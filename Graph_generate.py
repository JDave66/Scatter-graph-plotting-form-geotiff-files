# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:29:33 2023

@author: Jalpesh
"""
import rasterio
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

L2_Name = '20230420_L2.tif'
S3L2 = rasterio.open(L2_Name)  #LST for NDVI, LST, VZA, etc
# SLST_file = "20230107_SLST.tif"
RLST_file = "RLST_20230420.tif"
# SLST = S3L2.read(4)
CWV1 = S3L2.read(2)
cloud = S3L2.read(5)
SLST = S3L2.read(4)
cloud1 = cloud/255
CWV11 = CWV1/10

# Create a mask for cropland (assuming class 18 is cropland)
cloud_mask = cloud1 > 1
# Apply the mask to set corresponding values in SLST to NaN
CWV11[cloud_mask] = np.nan

# Set Times New Roman font and font size
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

SLST[SLST == 0] = np.nan
SLST[cloud_mask] = np.nan

# Open the two raster files using rasterio
with rasterio.open(RLST_file) as src2:
    # Read the raster data into numpy arrays
    arr2 = src2.read(1)
    arr2[arr2 == 0] = np.nan
    arr2[cloud_mask] = np.nan

    # Flatten the arrays into 1D arrays
    flat_arr1 = SLST.flatten()
    flat_arr2 = arr2.flatten()
    CWV = CWV11.flatten()

    # Remove any NaN or masked values from the arrays
    mask = np.logical_or(np.isnan(flat_arr1), np.isnan(flat_arr2))
    flat_arr1 = flat_arr1[~mask]
    flat_arr2 = flat_arr2[~mask]
    CWV = CWV[~mask]
    # Compute the R-square
    r2 = r2_score(flat_arr1, flat_arr2)
    total_elements = len(flat_arr1)

    # Compute the RMSE
    rmse = np.sqrt(mean_squared_error(flat_arr1, flat_arr2))
    
    # Compute the Mean Bias Error (MBE)
    mbe = np.mean(flat_arr1 - flat_arr2)
    mae_RF = mean_absolute_error(flat_arr1, flat_arr2)

    #finding the minimum and maximum value of X and Y
    min_val = min(min(flat_arr1), min(flat_arr2))
    max_val = max(max(flat_arr1), max(flat_arr2))

    # Create a scatter plot
    fig, ax = plt.subplots()
    # ax.scatter(flat_arr1, flat_arr2, alpha=0.5)
    scatter = plt.scatter(flat_arr1, flat_arr2, c=CWV, cmap='coolwarm', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Water Vapour in gm/cm²')
    plt.xlim(min_val,max_val)
    plt.ylim(min_val,max_val)
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', label='1:1 Line')
    ax.set_xlabel('SLSTR LST in K')
    ax.set_ylabel('Retrieved LST in K')
    # Set tick parameters
    ax.tick_params(axis='x', direction='in', pad=5)
    ax.tick_params(axis='y', direction='in', pad=5)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    text = f" N: {total_elements:.0f}\n R²: {r2:.2f}\n RMSE: {rmse:.2f} K\n MAE: {mae_RF:.2f} K"
    # bbox_props = dict(boxstyle='square', facecolor='white', edgecolor='black')
    ax.text(0.04, 0.94, text, transform=ax.transAxes, va='top', ha='left') #, bbox=bbox_props
    # Save the plot as a higher-resolution JPEG image
    L2_Na=L2_Name.split('.')[0]
    plt.savefig(f'scatter_plot_Color_CloudFree_{L2_Na}_new.jpg', dpi=800, format='jpeg')
    
    # Show the plot
    plt.show()