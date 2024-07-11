import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
import networks


   
    """" The code translates the images from digital domain to the P&S domain (vice-versa).
    The CycleGAN model architecture is loaded using the networks script.
    Cycle GAN simulators are loaded resulting in the translation of images.
    
    """
    
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def cyclegan(input_image):

    """"Image translation using trained CycleGAN simulators.

    Parameters:
        input_image (array) --  the input image array
    """
    #resize the input image to (256,256,3)
    input_image = cv2.resize(input_image, (256, 256), interpolation=cv2.INTER_CUBIC)
    # Convert BGR to RGB (OpenCV loads images in BGR by default)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # Convert to tensor and add batch dimension
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_nc = 3, 
    output_nc = 3,
    ngf = 64
    netG = 'unet_256'
    
    netG_A = networks.define_G(input_nc, output_nc, ngf, netG, 'instance',
                               not 'store_true', 'normal', 0.02, '0')

    #Load the trained Simulator
    model_path = 'latest_net_G_A.pth'
    netG_A.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
        generated_image = netG_A(input_tensor)
    # Convert the generated image tensor to a NumPy array
    generated_image = tensor2im(generated_image)
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
    return generated_image
