# MeronymNet
<!-- <img src='imgs/teaser_SBGAN.jpg' align="right" width=384> -->
<center><h2>A Hierarchical Model for Unified and Controllable Multi-Category Object Generation</h2></center>
<img src='https://github.com/meronymnet/meronymnet.github.io/blob/main/resources/meronymnet-overview-v2.png' align="center">
We introduce MeronymNet, a novel hierarchical approach for con- trollable, part-based generation of multi-category objects using a single unified model. We adopt a guided coarse-to-fine strategy involving semantically conditioned generation of bounding box layouts, pixel-level part layouts and ultimately, the object depic- tions themselves. We use Graph Convolutional Networks, Deep Recurrent Networks along with custom-designed Conditional Vari- ational Autoencoders to enable flexible, diverse and category-aware generation of 2-D objects in a controlled manner. The performance scores for generated objects reflect MeronymNet’s superior perfor- mance compared to multiple strong baselines and ablative variants.
<table align=center width=850px>
  <center><h1>Paper</h1></center>
  <tr>
  <td width=400px align=center>
  <!-- <p style="margin-top:4px;"></p> -->
  <a href="https://drive.google.com/file/d/1NnY4tcV1wnlSWMzT_Ae6hH6v5l8GCIrX/view?usp=sharing"><img style="height:200px" src="https://github.com/meronymnet/meronymnet.github.io/blob/main/resources/Paper_crop.png"/></a>
  <center>
  <span style="font-size:20pt"><a href="hhttps://drive.google.com/file/d/1NnY4tcV1wnlSWMzT_Ae6hH6v5l8GCIrX/view?usp=sharing">[Paper]</a>&nbsp;
  </center>
  </td>
  </tr>
  </table>
<center><h1>Code</h1></center>

## Prerequisites:
- NVIDIA GPU + CUDA CuDNN
- Python 3.6
- TensorFlow 1.15
- PyTorch 1.0
- Please install dependencies by
```
pip install -r requirements.txt
```

## Datasets
### PascalParts
- Access the dataset from <a href="http://roozbehm.info/pascal-parts/pascal-parts.html">PascalParts</a>
