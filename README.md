# Prediction of Supply and Use Tables based on Nonlinear Optimization

Implementation of nonlinear optimization methods for prediction of [Supply and Use Tables](https://en.wikipedia.org/wiki/Input%E2%80%93output_model) (SUTs) using historical data.

## Contents of the repository

* [MethodsFromArticle](MethodsFromArticle) — implementation of methods from the article ["Projection of Supply and Use Tables: Methods and their Empirical Assessment"](http://dx.doi.org/10.2139/ssrn.1539089) and the russian book ["Методы оптимизации и исследование операций для бакалавров информатики. Ч. II"](https://www.google.com/search?q=ISBN+978-5-89503-483-5)
* [methods_tester.ipynb](methods_tester.ipynb) - testing (applying) implemented methods using the tables from folder 'data'.
* [data](data) — SUTs dataset of the Netherlands for the fiscal years of 2010, 2011, and 2012 from the [WIOD Repository](http://www.wiod.org/)
* [data_rus](data_rus) — SUTs dataset of Russia for the fiscal years of 2012, 2013 from the [Rosstat Repository](https://gks.ru/)
* [presentation/main.pdf](presentation/main.pdf) — presentation of a brief description of the problem, mathematic model, testing model and results (in Russian)

## Requirements

* numpy >= 1.17.4
* scipy >= 1.3.1

## Installation

Copy the 'MethodsFromArticle' folder to your project folder.

## Example

```python
import numpy as np
from MethodsFromArticle import predict
from MethodsFromArticle import predict_grad

# Initial supply or use matrix
sup10 = np.load('data//sup10.npy')
# Constraint vectors — summation by rows and columns
sup11 = np.load('data//sup11.npy')
u = np.sum(sup11, axis=1)
v = np.sum(sup11, axis=0)

# Availabled methods for prediction:
#   INS - Improved Normalized Squared Difference
#   IWS - Improved Weighted Square Differences
#   ISD - Improved Square Differences
#   RAS - RAS method
pred_sup11 = predict(sup10, u, v, method='INS')

# Proximal gradient method
pred_sup11 = predict_grad(sup10, u, v)
```