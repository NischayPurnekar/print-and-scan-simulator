# Print and Scan(P&S) Simulator

This free software includes a sophisticated simulator designed to convert digital images into simulated print-and-scan versions and vice versa. Utilizing advanced image processing and machine learning, it accurately reproduces the typical variations, distortions, and artifacts associated with the print-scan process. It then converts these back into digital format, making it useful for image denoising and other related applications.

The unpaired image-to-image translation is achieved by training the CycleGAN architecture using Digital images and their corresponding Print and Scanned versions implemented in PyTorch.

# Trained Simulator Models

Available for download are the following models:

1) A simulator that converts images from the digital domain to the print and scan (P&S) domain is available [Google drive link] (https://drive.google.com/file/d/14RkAn_h2bqxFe-ey1mLqJ_t1Rq07Hf4E/view?usp=sharing)
2) A simulator that reverts images from the P&S domain back to the digital domain can be accessed [Google drive link] (https://drive.google.com/file/d/1QoDws1gGJ38vhW-souogWcbu_5H0IDm-/view?usp=sharing)

To run the Python script, follow these steps:

# Prerequisites
1) Ensure Python is installed: You’ll need Python 3.6 or later. You can check the version by running:

    python --version
3) Install required libraries: Install the necessary Python packages. You can do this in your environment by running:

   pip install torch torchvision numpy opencv-python
5) Have the networks module ready:
     The networks module (or networks.py file) should be in the same directory as your script, and it should contain the define_G function for loading the CycleGAN model.
     Ensure you also have the model file (from the Trained simulator models), latest_net_G_A.pth, in the working directory or adjust the path in the script if it's located elsewhere.
6) Prepare an input image
     Save an image file (e.g., input.jpg) in the same directory or specify its path in the code.

# Steps to run the script
1) Save the script as a Python file
2) Update the image path: If the __name__=="__main__": section, replace "path_to_input_image.jpg" with the actual path of your input image, like "input.jpg".
3) Run the script: Open a terminal or command prompt, navigate to the directory where the script is saved, and run the command:
     python test.py
4) Check the output: After running the script, the translated image will be saved as translated_image.jpg in the same directory. Open it to see the CycleGAN-translated result.

# Citation
If you utilize the simulators in your research, please cite our paper

@inproceedings{10.1145/3658664.3659635,
author = {Purnekar, Nischay and Abady, Lydia and Tondi, Benedetta and Barni, Mauro},
title = {Improving the Robustness of Synthetic Images Detection by Means of Print and Scan Augmentation},
year = {2024}
}
