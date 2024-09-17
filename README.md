# Differentiable Largest Connected Component (LCC) Layer

Official implementation of the ICANN 2024 paper Differentiable Largest Connected Component Layer for Image Matting. 

<p align="middle">
    <img src="illustration.png">
</p>

## Installation
Please install the LCC layer as follows.
```
python setup.py build_ext --inplace
```

## How to Use the Code
Original alpha matte | Processed alpha matte
--- | ---
![](alpha_matte_example.png) | ![](alpha_matte_processed_example.png)

This can be achieved by running
```
python example.py
```

## Future Work
By processing the mask of each channel, the LCC layer can be simply applied to other tasks, e.g., semantic segmentation.

