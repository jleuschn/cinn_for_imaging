import numpy as np 
import matplotlib.pyplot as plt 

import json 


with open('multi_scale_normal_num_images=100_psnrs.json', 'r') as fp:
    psnr_multi_scale_normal = json.load(fp)

with open("multi_scale_normal_num_images=100_ssims.json", 'r') as fp:
    ssim_multi_scale_normal = json.load(fp)

with open('multi_scale_radial_num_images=100_psnrs.json', 'r') as fp:
    psnr_multi_scale_radial = json.load(fp)

with open("multi_scale_radial_num_images=100_ssims.json", 'r') as fp:
    ssim_multi_scale_radial = json.load(fp)



psnr_multi_scale_normal_mean = [np.mean(psnr_multi_scale_normal[key]) for key in psnr_multi_scale_normal.keys()]
ssim_multi_scale_normal_mean = [np.mean(ssim_multi_scale_normal[key]) for key in ssim_multi_scale_normal.keys()]

psnr_multi_scale_radial_mean = [np.mean(psnr_multi_scale_radial[key]) for key in psnr_multi_scale_radial.keys()]
ssim_multi_scale_radial_mean = [np.mean(ssim_multi_scale_radial[key]) for key in ssim_multi_scale_radial.keys()]

num_samples = np.asarray([float(key) for key in psnr_multi_scale_normal.keys()])
print(num_samples)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.semilogx(num_samples[1:], psnr_multi_scale_normal_mean[1:], "o-", label="ms normal")
ax1.semilogx(num_samples[1:], psnr_multi_scale_radial_mean[1:], "o-", label="ms radial")
ax1.set_xlabel("number of samples")
ax1.set_ylabel("PSNR")
ax1.set_xlim(9, np.max(num_samples))

ax1.legend()


ax2.semilogx(num_samples[1:], ssim_multi_scale_normal_mean[1:], "o-")
ax2.semilogx(num_samples[1:], ssim_multi_scale_radial_mean[1:], "o-")
ax2.set_xlabel("number of samples")
ax2.set_ylabel("SSIM")
ax2.set_xlim(9, np.max(num_samples))

fig.suptitle("PSNR and SSIM by number of samples for cond. mean")

plt.show()