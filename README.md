
# Analyzing Gender Representation in Multilingual Models

This project includes the experiments described in the [paper](https://arxiv.org/pdf/2204.09168.pdf): 

**"Analyzing Gender Representation in Multilingual Models"** 

Hila Gonen, Shauli Ravfogel, Yoav Goldberg, RepL4NLP, 2022.

## Prerequisites

* Python 3
* Required INLP code is already in this repository

## Data

Please download relevant data files from this [folder](https://drive.google.com/drive/folders/17ON-wMI8RaYDhXFgOYD5N7gVw0LDqg7O?usp=sharing), save the data directory in parallel to the src directory.

The BiasBios data was crawled using the original scripts released by the creators of the BiosBias Dataset (De-Arteaga et al., 2019) and the Multilingual BiosBias Dataset (Zhao et al., 2020) - please refer to the paper for more details.
Note that we have removed the text itself from the data.
You may also download pretrained INLP projection matrices and additional data for reproducing the results easily and quickly.

### Details

## Extract representations for the datasets
For English, use **src/data/run_bias_bios.sh**
For French, use **src/data/run_bias_bios_fr.sh**
For Spanish, use **src/data/run_bias_bios_es.sh**

## Train INLP
Use the script **src/train_inlp.py**
Example:
```python train_inlp.py --lang EN --iters 300 --type avg --output_path <your-output-path> 
```
## Explained variance of PCA experiment (+ similarity between classifiers)

Use the script **src/pca_gender_repr_new.ipynb**.

## Gender Prediction Accuracy across Languages
Use the script **src/acc_across_langs_new.ipynb**

## Gender and profession classification
Use the scripts **src/classify_gender.py** and **src/classify_prof.py**

## Cite

If you find this project useful, please cite the paper:
```
@inproceedings{gonen_multigender22,
    title = "Analyzing Gender Representation in Multilingual Models",
    author = "Gonen, Hila and Ravfogel, Shauli and Goldberg, Yoav",
    booktitle = "Proceedings of the 7th Workshop on Representation Learning for NLP (RepL4NLP)",
    year = "2022",
}
```

## Contact

If you have any questions or suggestions, please contact [Hila Gonen](mailto:hilagnn@gmail.com).

## License

This project is licensed under Apache License - see the [LICENSE](LICENSE) file for details.

