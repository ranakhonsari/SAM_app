from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def sam_backend(image_np, input_point):
    """Run SAM model and return 3 segmentation masks with their scores."""
    
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = 'cpu'
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(image_np)

    # Get masks and scores
    masks, scores, _ = predictor.predict(
        point_coords=input_point, 
        point_labels=np.array([1]), 
        multimask_output=True
    )

    # Prepare masks as Base64
    masks_base64 = []
    for mask in masks:
        # Convert boolean mask to 0-255
        mask = mask.astype(np.uint8) * 255  

        # Convert to RGBA mask (Light blue with transparency)
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = 173  # R (light blue)
        mask_rgba[:, :, 1] = 216  # G (light blue)
        mask_rgba[:, :, 2] = 230  # B (light blue)
        mask_rgba[:, :, 3] = mask * 0.6  # Alpha (transparency)

        # Convert to PIL Image
        mask_image = Image.fromarray(mask_rgba, "RGBA")

        # Encode image to Base64
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        encoded_mask = base64.b64encode(buffered.getvalue()).decode("utf-8")
        masks_base64.append(encoded_mask)

    return masks_base64, scores.tolist()
