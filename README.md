# Generalized Contrastive Learning Beyond Regular Graph Data for Data-Driven Materials Discovery

Crystalline materials are essential components in a wide range of mechanical and electronic devices, and discovering materials with the desired properties plays a central role in advancing scientific and engineering systems. Each material has its unique crystal structure determining its physical properties (material properties), and capturing underlying relations between the crystal structures and the target material properties is the essential challenge in material design. However, crystal structures of the materials often exhibit non-smooth and discrete relations with their material properties, posing challenges for accurate graph-based machine learning. In this paper, we propose materials metric learning (MML) that is a generalized contrastive learning framework designed for crystal structures with continuous target values beyond lattice-structured image data with discrete class labels. MML generates latent graph representations strongly correlated to given material properties, facilitating both accurate prediction and generalization. In the experiments, we applied MML to downstream prediction models in material property prediction on extensive benchmark datasets, and the prediction models with MML achieved state-of-the-art prediction accuracy. Furthermore, as a real-world application of MML, we employ MML-based downstream prediction models for high-throughput screening to discover novel solar-cell materials. Our experimental results show that MML facilitates the identification of promising solar cell materials with the targeted properties.

* Paper: Gyoung S. Na, Generalized Contrastive Learning Beyond Regular Graph Data for Data-Driven Materials Discovery, xxxx xxxx.


## Dataset Sources
* **MPS-FE:** Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical review letters, 120(14), 145301.
* **MPS-PR:** Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical review letters, 120(14), 145301.
* **HOIP-GBG:**
* **HOIP-HBG:**
* **NLHM-GBG:**
* **MPL-FE:**
* **MPL-BGB:**
  

## Run
Execute `experiment.py` in each task folder. The embedding and prediction results are saved in the `results` folder.


## License
ACM acknowledges that this contribution was authored or co-authored by an employee, contractor or affiliate of a national government. As such, the Government retains a nonexclusive, royalty-free right to publish or reproduce this article, or to allow others to do so, for Government purposes only
