import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def unnormalize(tensor, mean, std):
    """
    Unnormalize a tensor image with mean and standard deviation.
    
    Args:
    - tensor (Tensor): The image tensor to unnormalize.
    - mean (list): The mean used to normalize the image.
    - std (list): The standard deviation used to normalize the image.
    
    Returns:
    - Tensor: The unnormalized image tensor.
    """
    # Clone the tensor to avoid changes to the original tensor
    tensor = tensor.clone()
    device = tensor.device
    
    # The mean and std have to be reshaped to [C, 1, 1] to broadcast properly
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    # Perform the reverse of normalization
    if tensor.is_cuda:
        mean = mean.to(device)
        std = std.to(device)
    tensor.mul_(std).add_(mean)  # Reverse the normalization
    
    return tensor

def overlay_keypoints_on_image(img, joints, maxvals):
    img = np.array(img)
    # Generate rainbow colors
    colors = plt.get_cmap('hsv')(np.linspace(0, 1, len(joints)))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    colors = tuple(colors.tolist())
    
    JOINTS= ["Left_eye_L", 
        "Left_eye_R", 
        "Left_eye", 
        "Right_eye_L", 
        "Right_eye_R", 
        "Right_eye", 
        "Nose", 
        "Lip_R", 
        "Lip_L", 
        "Lip_Up", 
        "Lip_Down", 
        "Left_ear", 
        "Right_ear"]
    
    joint2color = dict(zip(JOINTS, colors))
    
    # Create figure with two subplots
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: Image with joints
    img_with_joints = img.copy()
    img_with_joints = cv2.cvtColor(img_with_joints, cv2.COLOR_BGR2RGB)
    text_content = []
    
    for i, (joint, maxval) in enumerate(zip(joints, maxvals)):
        x, y = joint
        color = joint2color[JOINTS[i]]
        img_with_joints = cv2.circle(img_with_joints, (int(x), int(y)), 3, color, 2)
        joint_text = f"{JOINTS[i].upper()}: {str(maxval[0])[:4]}"
        text_content.append((joint_text, [c/255 for c in color]))
        
    ax_img.imshow(img_with_joints)
    ax_img.set_title('')
    ax_img.axis('off')
    
    # Right subplot: Text information
    ax_text.axis('off')
    # Display text with matching colors
    y_position = 0.95
    for text, color in text_content:
        ax_text.text(0, y_position, text,
                    transform=ax_text.transAxes,
                    color=color,
                    fontfamily='monospace',
                    fontsize=13,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        y_position -= 0.05
    
    ax_text.set_title('')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close figure to free memory
    plt.close(fig)
    
    return plot_img

