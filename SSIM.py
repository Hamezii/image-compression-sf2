import numpy as np

def calculate_ssim(image1, image2):
    # Compute mean, variance, and covariance
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    sigma1 = np.std(image1)
    sigma2 = np.std(image2)
    cov = np.cov(image1.flatten(), image2.flatten())[0, 1]

    # Constants for stability
    k1 = 0.01
    k2 = 0.03
    L = 255  # Maximum pixel value

    # Compute SSIM
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))

    return ssim

# # Create two example 256x256 ndarrays (replace with your own images)
# image1 = np.random.randint(0, 256, size=(256, 256))
# image2 = np.random.randint(0, 256, size=(256, 256))

# # Calculate SSIM
# ssim_score = calculate_ssim(image1, image2)

# print(f"SSIM between the images: {ssim_score:.4f}")