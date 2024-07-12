# Print and Scan(P&S) Simulator

This free software includes a sophisticated simulator designed to convert digital images into simulated print-and-scan versions and vice versa. Utilizing advanced image processing and machine learning, it accurately reproduces the typical variations, distortions, and artifacts associated with the print-scan process. It then converts these back into digital format, making it useful for image denoising and other related applications.

The unpaired image-to-image translation is achieved by training the CycleGAN architecture using Digital images and their corresponding Print and Scanned versions implemented in PyTorch.

# Trained Simulator Models

Available for download are the following models:

1) A simulator that converts images from the digital domain to the print and scan (P&S) domain is available [Google drive link] (https://drive.google.com/file/d/14RkAn_h2bqxFe-ey1mLqJ_t1Rq07Hf4E/view?usp=sharing)
2) A simulator that reverts images from the P&S domain back to the digital domain can be accessed [Google drive link] (https://drive.google.com/file/d/1QoDws1gGJ38vhW-souogWcbu_5H0IDm-/view?usp=sharing)

# Citation
If you utilize the simulators in your research, please cite our paper

@inproceedings{10.1145/3658664.3659635,
author = {Purnekar, Nischay and Abady, Lydia and Tondi, Benedetta and Barni, Mauro},
title = {Improving the Robustness of Synthetic Images Detection by Means of Print and Scan Augmentation},
year = {2024}
}
