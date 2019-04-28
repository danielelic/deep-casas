# deep-casas

Deep learning and LSTM approaches for human activity recognition.

This project has been included in the paper "A Sequential Deep Learning Application for Recognising Human Activities in Smart Homes" accepted for Neurocomputing journal.

If you find this code useful, we encourage you to cite the paper. BibTeX:


```
@article{liciotti2019sequential,
    author   = "Daniele Liciotti and Michele Bernardini and Luca Romeo and Emanuele Frontoni",
    title    = "A Sequential Deep Learning Application for Recognising Human Activities in Smart Homes",
    journal  = "Neurocomputing",
    year     = "2019",
    issn     = "0925-2312",
    doi      = "https://doi.org/10.1016/j.neucom.2018.10.104",
    url      = "http://www.sciencedirect.com/science/article/pii/S0925231219304862",
    keywords = "Smart Home, Human Activity Recognition, Deep Learning, LSTM",
    abstract = "The recent advancement and development of computer electronic devices has led to the adoption of smart home sensing systems, stimulating the demand for associated products and services. Accordingly, the increasingly large amount of data calls the machine learning (ML) field for automatic recognition of human behaviour. In this work, different deep learning (DL) models that learn to classify human activities were proposed. In particular, the long short-term memory (LSTM) was applied for modelling spatio-temporal sequences acquired by smart home sensors. Experimental results performed on the Center for Advanced Studies in Adaptive Systems datasets show that the proposed LSTM-based approaches outperform existing DL and ML methods, giving superior results compared to the existing literature."
}
```

The code has been tested on:

* Python 3.5.2
* Keras 2.1.5
* TensorFlow 1.3.0

## Data

The `data.py` script loads some [CASAS datasets](http://casas.wsu.edu/datasets/) and saves them into NumPy binary format files `.npy` for faster loading later.
```
python data.py
```

## Train

```
python train.py --v LSTM
python train.py --v biLSTM
python train.py --v Ensemble2LSTM
python train.py --v CascadeEnsembleLSTM
python train.py --v CascadeLSTM
```

## Authors

* Daniele Liciotti | [GitHub](https://github.com/danielelic)
* Michele Bernardini
* Luca Romeo
