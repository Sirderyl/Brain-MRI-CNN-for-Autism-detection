# The interplay between neonatal brain structure and function

This is a research project conducted as a dissertation project for a Computer Science degree at Northumbria University. The dataset consists of structural and functional brain MRI scans of term-born and preterm-born neonates along with their corresponding psychological autism assessment conducted when the subjects reached 18 months of age. The goal of this project was to identify whether neural networks could be employed to detect autism from biological markers in the brain structure. Diagnosing autism at birth could have a positive impact on the support the newborns receive, which in turn could improve their quiality of life.

For demonstration, only small sample of the MRI data is included in this repository, as the dataset contains over 800 subjects and takes up over 1 TB of storage.

The dataset and more details about the data can be found [here](https://www.developingconnectome.org).

The dissertation paper with the results can be found [here](The%20interplay%20between%20neonatal%20brain%20structure%20and%20function.pdf).

To train the pre-trained 2D CNN model on the MRI, run the `CNN_2D_train.py` script.  
To train the custom 3D CNN model on the MRI, run the `CNN_3D_train.py` script.  
To train the custom 3D CNN model with oversampling on the MRI, run the `CNN_3D_train_oversampling.py` script.  

All the CSV files required for training are already provided. If you wish to create them from scratch, follow these steps:  
1. Run the `tor_script.sh` script (this may take around 15 minutes)  
2. Run all cells in the `csv_create.ipynb` script

&copy; Filip Kovarik, Northumbria University 2024