<<<<<<< HEAD
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from skimage import io, transform
from skimage.transform import resize
from PIL import Image


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from hierarchialdet.predictor import VisualizationDemo
from hierarchialdet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from hierarchialdet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from ultralytics import YOLO
import numpy as np

def merge_yolo_mask_with_original(yolo_mask, original_image_shape, bbox):
    """
    Resizes a YOLO mask for a cropped image and places it back onto a mask
    of the original image's dimensions.

    Args:
        yolo_mask (np.ndarray): The binary mask from YOLO (e.g., 640x640).
        original_image_shape (tuple): The shape of the original image (height, width).
        bbox (tuple): The bounding box coordinates in (x_min, y_min, x_max, y_max) format.

    Returns:
        np.ndarray: A new mask with the same dimensions as the original image,
                    containing the YOLO prediction in the correct location.
    """
    # 1. Get bbox coordinates and original image dimensions
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    orig_h, orig_w = original_image_shape

    # 2. Get the dimensions of the cropped area
    crop_h = y_max - y_min
    crop_w = x_max - x_min
      # 3. CONTROLLO DI VALIDITÀ PER EVITARE ValueError
    if crop_w <= 0 or crop_h <= 0:
        print(f"AVVISO: Bounding box non valida o zero-sized: ({x_min}, {y_min}, {x_max}, {y_max}). Ritorno maschera vuota.")
        # Restituisce una maschera vuota con le dimensioni dell'originale
        return np.zeros((orig_h, orig_w), dtype=np.uint8)


#     # 3. Create a blank mask with the original image's dimensions
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

#     # 4. Resize the YOLO mask to the dimensions of the cropped area
#     # Note: OpenCV expects (width, height)
    resized_yolo_mask = cv2.resize(
        yolo_mask,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )
    print(resized_yolo_mask.shape)
    # 5. Place the resized YOLO mask back onto the full mask
    full_mask[y_min:y_max, x_min:x_max] = resized_yolo_mask

    return full_mask

def calculate_dice(ground_truth, segmentation):
    """
    Calcola il Dice Similarity Coefficient (DSC) tra due maschere binarie.

    Args:
        ground_truth (np.ndarray): Maschera di ground truth (binaria).
        segmentation (np.ndarray): Maschera di segmentazione (binaria).

    Returns:
        float: Il valore del DSC.
    """

    # Trasformare le maschere in array booleani
    ground_truth = ground_truth.astype(bool)
    segmentation = segmentation.astype(bool)


    # Calcolare l'intersezione (AND logico)
    intersection = np.sum(ground_truth & segmentation)

    # Calcolare l'area di ciascuna maschera
    area_ground_truth = np.sum(ground_truth)
    area_segmentation = np.sum(segmentation)

    # Evitare la divisione per zero nel caso di maschere vuote
    if area_ground_truth == 0 and area_segmentation == 0:
        return 1.0  # Se entrambe sono vuote, la sovrapposizione è perfetta.
    
    # Calcolare il DSC
    dice = (2.0 * intersection) / (area_ground_truth + area_segmentation)
    
    return dice


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    
    parser.add_argument(
        "--nclass",
        type=int,
        default=1,
        help="Number of trained classes",
    )
    parser.add_argument(
        "--seg_model",
        type=str,
        help="Segmentation model: yolo , medsam",
        default="medsam",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
def merge_yolo_mask_with_original(yolo_mask, original_image_shape, bbox):
    """
    Resizes a YOLO mask for a cropped image and places it back onto a mask
    of the original image's dimensions.

    Args:
        yolo_mask (np.ndarray): The binary mask from YOLO (e.g., 640x640).
        original_image_shape (tuple): The shape of the original image (height, width).
        bbox (tuple): The bounding box coordinates in (x_min, y_min, x_max, y_max) format.

    Returns:
        np.ndarray: A new mask with the same dimensions as the original image,
                    containing the YOLO prediction in the correct location.
    """
    # 1. Get bbox coordinates and original image dimensions
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    orig_h, orig_w = original_image_shape

    # 2. Get the dimensions of the cropped area
    crop_h = y_max - y_min
    crop_w = x_max - x_min

     # 3. CONTROLLO DI VALIDITÀ PER EVITARE ValueError
    if crop_w <= 0 or crop_h <= 0:
        print(f"AVVISO: Bounding box non valida o zero-sized: ({x_min}, {y_min}, {x_max}, {y_max}). Ritorno maschera vuota.")
        # Restituisce una maschera vuota con le dimensioni dell'originale
        return np.zeros((orig_h, orig_w), dtype=np.uint8)


    # 3. Create a blank mask with the original image's dimensions
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # 4. Resize the YOLO mask to the dimensions of the cropped area
    # Note: OpenCV expects (width, height)
    resized_yolo_mask = cv2.resize(
        yolo_mask,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )
    print(resized_yolo_mask.shape)
    # 5. Place the resized YOLO mask back onto the full mask
    full_mask[y_min:y_max, x_min:x_max] = resized_yolo_mask

    return full_mask

# constants
WINDOW_NAME = "Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images"

colorMap = {
            0: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            1: (1.0, 0.4980392156862745, 0.054901960784313725),
            2: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            3: (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            4: (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            5: (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
            }
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, k=(args.nclass-1))

    OUTPUT_DIR = args.output
    if args.input:
        assert args.seg_model in ["yolo","medsam"], "Select a segmenation model, yolo or madsam; for --seg_model"
        if args.seg_model == "medsam":
            medsam_model = sam_model_registry["vit_b"](checkpoint='weights/MedSAM/final.pth')
            medsam_model = medsam_model.to('cuda:0')
            medsam_model.eval()
        if args.seg_model == "yolo":
             model = YOLO("weights/yolo-seg/best.pt")
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            print(path)
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            # --- PREPARAZIONE BASE ---
            image_base_rgb = visualized_output.get_image()  # RGB
            image_base_bgr = image_base_rgb[:, :, ::-1].copy()  # Converti in BGR per cv2
            overlay = image_base_bgr.copy()
            ALFA = 0.5  # Trasparenza delle maschere
            boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy().astype(int)
            classes = [int(x) for x in predictions['instances'].pred_classes_3]
            base_filename = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path)
            img_originale = Image.open(path).convert("RGB")
            img_originale_np = np.array(img_originale)
            W_im,H_im = img.size
            results_segmentation = []
            maschera_aggregata_nera = np.zeros((H_im, W_im), dtype=np.uint8)

            img_np = io.imread(path)
            img_3c = img_np
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to('cuda:0')
            )
            if args.seg_model == "medsam":
                for i, ((x1, y1, x2, y2),cls) in enumerate(zip(boxes,classes)):
                
                    boxnp = np.array([[int(x) for x in (x1, y1, x2, y2)]]) 
                    box_1024 = boxnp * 1024 / np.array([W, H, W, H])
                    with torch.no_grad():
                        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

                    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
                    
                    resized_medsam_seg = resize(medsam_seg,(1024, 1024), anti_aliasing=True)
                    resized_medsam_seg = (resized_medsam_seg > 0).astype(np.uint8)

                    try:
                        maschera_singola_nera = np.zeros((H, W), dtype=np.uint8)
                        maschera_singola_nera = medsam_seg
                        if args.output:
                            mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_mask_{i}_nero.png")
                            cv2.imwrite(mask_path, maschera_singola_nera * 255) # Moltiplichiamo per 255 per salvarla come immagine bianca su nero
                        maschera_aggregata_nera = np.maximum(maschera_aggregata_nera, maschera_singola_nera)
                        # 
                        maschera_bool = medsam_seg.astype(bool)
                        colore_rgb = colorMap[cls]
                        colore_bgr = tuple(int(255*c) for c in colore_rgb[::-1])
                        overlay[maschera_bool] = colore_bgr
                    except Exception as e:
                        print(f'Errore durante l\'elaborazione della maschera {i}: {e}')
                maschera_aggregata_bool = maschera_aggregata_nera.astype(bool)

                immagine_mascherata_nera = np.zeros_like(img_originale_np)
                immagine_mascherata_nera[maschera_aggregata_bool] = img_originale_np[maschera_aggregata_bool]
                if args.output:
                    final_mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_MASCHERA_AGGREGATA_nero.png")
                    cv2.imwrite(final_mask_path, maschera_aggregata_nera * 255)
                if args.output:
                    final_masked_path = os.path.join(OUTPUT_DIR, f"{base_filename}_AGGREGATA_su_immagine.png")
                    Image.fromarray(immagine_mascherata_nera).save(final_masked_path)
                print(visualized_output)            
                final_output_bgr = cv2.addWeighted(overlay, ALFA, image_base_bgr, 1 - ALFA, 0)

            if args.seg_model == "yolo":
                for i, ((x1, y1, x2, y2),cls) in enumerate(zip(boxes,classes)):

                    cropped_img = img.crop((x1, y1, x2, y2) )
                    results = model(cropped_img)  
                    yolo_masks = results[0].masks 
                    if yolo_masks is not None:
                        try:
                            yolo_seg_mask = yolo_masks.data[0].cpu().numpy()
                            yolo_seg_mask = (yolo_seg_mask > 0).astype(np.uint8)
                        except:
                            print('errore!!!!')
                        H,W = cropped_img.size
                        resized_yolo_mask_skimage = merge_yolo_mask_with_original(
                                yolo_seg_mask,
                                (H_im, W_im),
                                (int(x1), int(y1), int(x2), int(y2))
                            )
                        resized_yolo_mask_skimage = (resized_yolo_mask_skimage > 0).astype(np.uint8)
                        try:
                            maschera_singola_nera = np.zeros((H, W), dtype=np.uint8)
                            maschera_singola_nera = resized_yolo_mask_skimage
                            if args.output:
                                mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_mask_{i}_nero.png")
                                cv2.imwrite(mask_path, maschera_singola_nera * 255) 
                            
                            maschera_aggregata_nera = np.maximum(maschera_aggregata_nera, maschera_singola_nera)
                            maschera_bool = resized_yolo_mask_skimage.astype(bool)
                            colore_rgb = colorMap[cls]
                            colore_bgr = tuple(int(255*c) for c in colore_rgb[::-1])
                            overlay[maschera_bool] = colore_bgr
                        except Exception as e:
                            print(f'Errore durante l\'elaborazione della maschera {i}: {e}')
                if args.output:
                    final_mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_MASCHERA_AGGREGATA_nero.png")
                    cv2.imwrite(final_mask_path, maschera_aggregata_nera * 255)

                maschera_aggregata_bool = maschera_aggregata_nera.astype(bool)

                immagine_mascherata_nera = np.zeros_like(img_originale_np)
                immagine_mascherata_nera[maschera_aggregata_bool] = img_originale_np[maschera_aggregata_bool]
                if args.output:
                    final_masked_path = os.path.join(OUTPUT_DIR, f"{base_filename}_AGGREGATA_su_immagine.png")
                    Image.fromarray(immagine_mascherata_nera).save(final_masked_path)
                
                final_output_bgr = cv2.addWeighted(overlay, ALFA, image_base_bgr, 1 - ALFA, 0)

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                Image.fromarray(final_output_bgr).save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, final_output_bgr)
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
=======
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from skimage import io, transform
from skimage.transform import resize
from PIL import Image


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from hierarchialdet.predictor import VisualizationDemo
from hierarchialdet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from hierarchialdet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from ultralytics import YOLO
import numpy as np

def merge_yolo_mask_with_original(yolo_mask, original_image_shape, bbox):
    """
    Resizes a YOLO mask for a cropped image and places it back onto a mask
    of the original image's dimensions.

    Args:
        yolo_mask (np.ndarray): The binary mask from YOLO (e.g., 640x640).
        original_image_shape (tuple): The shape of the original image (height, width).
        bbox (tuple): The bounding box coordinates in (x_min, y_min, x_max, y_max) format.

    Returns:
        np.ndarray: A new mask with the same dimensions as the original image,
                    containing the YOLO prediction in the correct location.
    """
    # 1. Get bbox coordinates and original image dimensions
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    orig_h, orig_w = original_image_shape

    # 2. Get the dimensions of the cropped area
    crop_h = y_max - y_min
    crop_w = x_max - x_min
      # 3. CONTROLLO DI VALIDITÀ PER EVITARE ValueError
    if crop_w <= 0 or crop_h <= 0:
        print(f"AVVISO: Bounding box non valida o zero-sized: ({x_min}, {y_min}, {x_max}, {y_max}). Ritorno maschera vuota.")
        # Restituisce una maschera vuota con le dimensioni dell'originale
        return np.zeros((orig_h, orig_w), dtype=np.uint8)


#     # 3. Create a blank mask with the original image's dimensions
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

#     # 4. Resize the YOLO mask to the dimensions of the cropped area
#     # Note: OpenCV expects (width, height)
    resized_yolo_mask = cv2.resize(
        yolo_mask,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )
    print(resized_yolo_mask.shape)
    # 5. Place the resized YOLO mask back onto the full mask
    full_mask[y_min:y_max, x_min:x_max] = resized_yolo_mask

    return full_mask

def calculate_dice(ground_truth, segmentation):
    """
    Calcola il Dice Similarity Coefficient (DSC) tra due maschere binarie.

    Args:
        ground_truth (np.ndarray): Maschera di ground truth (binaria).
        segmentation (np.ndarray): Maschera di segmentazione (binaria).

    Returns:
        float: Il valore del DSC.
    """

    # Trasformare le maschere in array booleani
    ground_truth = ground_truth.astype(bool)
    segmentation = segmentation.astype(bool)


    # Calcolare l'intersezione (AND logico)
    intersection = np.sum(ground_truth & segmentation)

    # Calcolare l'area di ciascuna maschera
    area_ground_truth = np.sum(ground_truth)
    area_segmentation = np.sum(segmentation)

    # Evitare la divisione per zero nel caso di maschere vuote
    if area_ground_truth == 0 and area_segmentation == 0:
        return 1.0  # Se entrambe sono vuote, la sovrapposizione è perfetta.
    
    # Calcolare il DSC
    dice = (2.0 * intersection) / (area_ground_truth + area_segmentation)
    
    return dice


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    
    parser.add_argument(
        "--nclass",
        type=int,
        default=1,
        help="Number of trained classes",
    )
    parser.add_argument(
        "--seg_model",
        type=str,
        help="Segmentation model: yolo , medsam",
        default="medsam",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
def merge_yolo_mask_with_original(yolo_mask, original_image_shape, bbox):
    """
    Resizes a YOLO mask for a cropped image and places it back onto a mask
    of the original image's dimensions.

    Args:
        yolo_mask (np.ndarray): The binary mask from YOLO (e.g., 640x640).
        original_image_shape (tuple): The shape of the original image (height, width).
        bbox (tuple): The bounding box coordinates in (x_min, y_min, x_max, y_max) format.

    Returns:
        np.ndarray: A new mask with the same dimensions as the original image,
                    containing the YOLO prediction in the correct location.
    """
    # 1. Get bbox coordinates and original image dimensions
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    orig_h, orig_w = original_image_shape

    # 2. Get the dimensions of the cropped area
    crop_h = y_max - y_min
    crop_w = x_max - x_min

     # 3. CONTROLLO DI VALIDITÀ PER EVITARE ValueError
    if crop_w <= 0 or crop_h <= 0:
        print(f"AVVISO: Bounding box non valida o zero-sized: ({x_min}, {y_min}, {x_max}, {y_max}). Ritorno maschera vuota.")
        # Restituisce una maschera vuota con le dimensioni dell'originale
        return np.zeros((orig_h, orig_w), dtype=np.uint8)


    # 3. Create a blank mask with the original image's dimensions
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # 4. Resize the YOLO mask to the dimensions of the cropped area
    # Note: OpenCV expects (width, height)
    resized_yolo_mask = cv2.resize(
        yolo_mask,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )
    print(resized_yolo_mask.shape)
    # 5. Place the resized YOLO mask back onto the full mask
    full_mask[y_min:y_max, x_min:x_max] = resized_yolo_mask

    return full_mask

# constants
WINDOW_NAME = "Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images"

colorMap = {
            0: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            1: (1.0, 0.4980392156862745, 0.054901960784313725),
            2: (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            3: (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            4: (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            5: (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
            }
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, k=(args.nclass-1))

    OUTPUT_DIR = args.output
    if args.input:
        assert args.seg_model in ["yolo","medsam"], "Select a segmenation model, yolo or madsam; for --seg_model"
        if args.seg_model == "medsam":
            medsam_model = sam_model_registry["vit_b"](checkpoint='weights/MedSAM/final.pth')
            medsam_model = medsam_model.to('cuda:0')
            medsam_model.eval()
        if args.seg_model == "yolo":
             model = YOLO("weights/yolo-seg/best.pt")
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            print(path)
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            # --- PREPARAZIONE BASE ---
            image_base_rgb = visualized_output.get_image()  # RGB
            image_base_bgr = image_base_rgb[:, :, ::-1].copy()  # Converti in BGR per cv2
            overlay = image_base_bgr.copy()
            ALFA = 0.5  # Trasparenza delle maschere
            boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy().astype(int)
            classes = [int(x) for x in predictions['instances'].pred_classes_3]
            base_filename = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path)
            img_originale = Image.open(path).convert("RGB")
            img_originale_np = np.array(img_originale)
            W_im,H_im = img.size
            results_segmentation = []
            maschera_aggregata_nera = np.zeros((H_im, W_im), dtype=np.uint8)

            img_np = io.imread(path)
            img_3c = img_np
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to('cuda:0')
            )
            if args.seg_model == "medsam":
                for i, ((x1, y1, x2, y2),cls) in enumerate(zip(boxes,classes)):
                
                    boxnp = np.array([[int(x) for x in (x1, y1, x2, y2)]]) 
                    box_1024 = boxnp * 1024 / np.array([W, H, W, H])
                    with torch.no_grad():
                        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

                    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
                    
                    resized_medsam_seg = resize(medsam_seg,(1024, 1024), anti_aliasing=True)
                    resized_medsam_seg = (resized_medsam_seg > 0).astype(np.uint8)

                    try:
                        maschera_singola_nera = np.zeros((H, W), dtype=np.uint8)
                        maschera_singola_nera = medsam_seg
                        if args.output:
                            mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_mask_{i}_nero.png")
                            cv2.imwrite(mask_path, maschera_singola_nera * 255) # Moltiplichiamo per 255 per salvarla come immagine bianca su nero
                        maschera_aggregata_nera = np.maximum(maschera_aggregata_nera, maschera_singola_nera)
                        # 
                        maschera_bool = medsam_seg.astype(bool)
                        colore_rgb = colorMap[cls]
                        colore_bgr = tuple(int(255*c) for c in colore_rgb[::-1])
                        overlay[maschera_bool] = colore_bgr
                    except Exception as e:
                        print(f'Errore durante l\'elaborazione della maschera {i}: {e}')
                maschera_aggregata_bool = maschera_aggregata_nera.astype(bool)

                immagine_mascherata_nera = np.zeros_like(img_originale_np)
                immagine_mascherata_nera[maschera_aggregata_bool] = img_originale_np[maschera_aggregata_bool]
                if args.output:
                    final_mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_MASCHERA_AGGREGATA_nero.png")
                    cv2.imwrite(final_mask_path, maschera_aggregata_nera * 255)
                if args.output:
                    final_masked_path = os.path.join(OUTPUT_DIR, f"{base_filename}_AGGREGATA_su_immagine.png")
                    Image.fromarray(immagine_mascherata_nera).save(final_masked_path)
                print(visualized_output)            
                final_output_bgr = cv2.addWeighted(overlay, ALFA, image_base_bgr, 1 - ALFA, 0)

            if args.seg_model == "yolo":
                for i, ((x1, y1, x2, y2),cls) in enumerate(zip(boxes,classes)):

                    cropped_img = img.crop((x1, y1, x2, y2) )
                    results = model(cropped_img)  
                    yolo_masks = results[0].masks 
                    if yolo_masks is not None:
                        try:
                            yolo_seg_mask = yolo_masks.data[0].cpu().numpy()
                            yolo_seg_mask = (yolo_seg_mask > 0).astype(np.uint8)
                        except:
                            print('errore!!!!')
                        H,W = cropped_img.size
                        resized_yolo_mask_skimage = merge_yolo_mask_with_original(
                                yolo_seg_mask,
                                (H_im, W_im),
                                (int(x1), int(y1), int(x2), int(y2))
                            )
                        resized_yolo_mask_skimage = (resized_yolo_mask_skimage > 0).astype(np.uint8)
                        try:
                            maschera_singola_nera = np.zeros((H, W), dtype=np.uint8)
                            maschera_singola_nera = resized_yolo_mask_skimage
                            if args.output:
                                mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_mask_{i}_nero.png")
                                cv2.imwrite(mask_path, maschera_singola_nera * 255) 
                            
                            maschera_aggregata_nera = np.maximum(maschera_aggregata_nera, maschera_singola_nera)
                            maschera_bool = resized_yolo_mask_skimage.astype(bool)
                            colore_rgb = colorMap[cls]
                            colore_bgr = tuple(int(255*c) for c in colore_rgb[::-1])
                            overlay[maschera_bool] = colore_bgr
                        except Exception as e:
                            print(f'Errore durante l\'elaborazione della maschera {i}: {e}')
                if args.output:
                    final_mask_path = os.path.join(OUTPUT_DIR, f"{base_filename}_MASCHERA_AGGREGATA_nero.png")
                    cv2.imwrite(final_mask_path, maschera_aggregata_nera * 255)

                maschera_aggregata_bool = maschera_aggregata_nera.astype(bool)

                immagine_mascherata_nera = np.zeros_like(img_originale_np)
                immagine_mascherata_nera[maschera_aggregata_bool] = img_originale_np[maschera_aggregata_bool]
                if args.output:
                    final_masked_path = os.path.join(OUTPUT_DIR, f"{base_filename}_AGGREGATA_su_immagine.png")
                    Image.fromarray(immagine_mascherata_nera).save(final_masked_path)
                
                final_output_bgr = cv2.addWeighted(overlay, ALFA, image_base_bgr, 1 - ALFA, 0)

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                Image.fromarray(final_output_bgr).save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, final_output_bgr)
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
>>>>>>> 49ca027d0ce94746b82bf9ab20198f1187435c9e
