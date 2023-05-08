## Final Project: Applied Unsupervised Learning
#### Project name: Market Segmentation in Insurance
#### Author: Qichun Yu

## Table of Contents
1. [Introduction](#Abstract)  
    1.1. [Abstract](#Abstract)  
    1.2. [Use Case](#Use-Case)  
    1.3. [Load and Read Data](#Load-and-Read-Data)  
    1.4. [Data Cleaning](#Data-Cleaning)
2. [Analysis and Visualization](#Analysis-and-Visualization)
3. [Preprocessing](#Preprocessing)
4. [Models](#Models)  
    4.1. [K-means Clustering](#K-means-Clustering)  
    4.2. [Hierarchical Clustering](#Hierarchical-Clustering)  
    4.3. [Density-Based Spatial Clustering of Applications with Noise (DBSCAN)](#Density-Based-Spatial-Clustering-of-Applications-with-Noise-(DBSCAN))  
5. [Dimensionality Reduction (PCA)](#Dimensionality-Reduction)  
    5.1. [PCA K-means](#PCA-K-means)    
    5.2. [PCA Hierarchical Clustering](#PCA-Hierarchical-Clustering)  
    5.3. [PCA DBSCAN](#PCA-DBSCAN)  
6. [Dimensionality Reduction (TSNE)](#T-Distributed-Stochastic-Neighbor-Embedding-(TSNE))  
    6.1. [TSNE K-means](#TSNE-K-means)    
    6.2. [TSNE Hierarchical Clustering](#TSNE-Hierarchical-Clustering)  
    6.3. [TSNE DBSCAN](#TSNE-DBSCAN)  
7. [Discussion](#Discussion)
8. [Conclusion](#Conclusion)

### Abstract

The project is developing **unsupervised machine-learning models** to group customers into segments for the purpose to give insurance product recommendations. Customers are divided into subgroups based on some types of similar characteristics. The dataset includes summary information on 18 behavioural variables from the 8,950 active credit cardholders. Behaviours include how a customer spends and pays over time. The notebook explores different unsupervised algorithms such as **k-means, hierarchical clustering, and DBSCAN** for an insurance company to divide customers into groups to optimize marketing campaigns for insurance products. Standardization is used to rescale data to have a mean of 0 and a standard deviation of 1.  **PCA and TSNE** methods are used for dimensionality reduction and visualization.  After comparing with the silhouette score and visualized plots, the optimal model is the k-means method with a k value of three that is trained with PCA scaled data. There are small groups of people who have similar behaviours on purchasing, cash advances, credit limits and so on. The K-means clustering method helps identify the group that has similar features. After the segmentation, an insurance company will provide insurance product recommendations based on their characteristics. 

### Use Case
The insurance industry is competitive. Building strong relationships with customers and maintaining customer engagement outside a claim or a renewal is important. An insurance company is developing a machine learning model to classify customers to provide recommendations on insurance products. Customer segmentation is dividing customers into different groups that have similar characteristics, needs, or goals. The insurance company can offer various products such as saving plans, loans, wealth management and so on to different segments. A successful machine learning model can help the company optimize marketing campaigns, identify new opportunities, and increase customer retention rates. 

### Dataset

The sample Dataset summarizes the usage behavior of about 8,950 active credit cardholders during the last 6 months. The file is at a customer level with 18 behavioral features:
<ol>
    <li>CustID</li>
<li>Balance</li>
<li>Balance Frequency</li>
<li>Purchases</li>
<li>One-off Purchases</li>
<li>Installment Purchases</li>
<li>Cash Advance</li>
<li>Purchases Frequency</li>
<li>One-off Purchases Frequency</li>
<li>Purchases Installments Frequency</li>
<li>Cash Advance Frequency</li>
<li>Cash Advance TRX</li>
<li>Purchases TRX</li>
<li>Credit Limit</li>
<li>Payments</li>
<li>Minimum Payments</li>
<li>PRC Full payment</li>
<li>Tenure</li>
</ol>

### Citation

Jillani Soft Tech.(September, 2022). Market Segmentation in Insurance Unsupervised. Retrieved from https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised.

### Import Modules


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# visualized Dendrogram
import scipy.cluster.hierarchy as sch 
%matplotlib inline
import seaborn as sns
# standardize Data
from sklearn.preprocessing import StandardScaler
# import libraries for unsupervised method
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

### Load and Read Data


```python
df = pd.read_csv("Customer Data.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C10001</td>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.4</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C10002</td>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C10003</td>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C10004</td>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C10005</td>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (8950, 18)



There are 8,950 rows with 18 columns. 


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8945</th>
      <td>C19186</td>
      <td>28.493517</td>
      <td>1.000000</td>
      <td>291.12</td>
      <td>0.00</td>
      <td>291.12</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>325.594462</td>
      <td>48.886365</td>
      <td>0.50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8946</th>
      <td>C19187</td>
      <td>19.183215</td>
      <td>1.000000</td>
      <td>300.00</td>
      <td>0.00</td>
      <td>300.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>275.861322</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8947</th>
      <td>C19188</td>
      <td>23.398673</td>
      <td>0.833333</td>
      <td>144.40</td>
      <td>0.00</td>
      <td>144.40</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0</td>
      <td>5</td>
      <td>1000.0</td>
      <td>81.270775</td>
      <td>82.418369</td>
      <td>0.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8948</th>
      <td>C19189</td>
      <td>13.457564</td>
      <td>0.833333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>36.558778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>2</td>
      <td>0</td>
      <td>500.0</td>
      <td>52.549959</td>
      <td>55.755628</td>
      <td>0.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8949</th>
      <td>C19190</td>
      <td>372.708075</td>
      <td>0.666667</td>
      <td>1093.25</td>
      <td>1093.25</td>
      <td>0.00</td>
      <td>127.040008</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>2</td>
      <td>23</td>
      <td>1200.0</td>
      <td>63.165404</td>
      <td>88.288956</td>
      <td>0.00</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8950 entries, 0 to 8949
    Data columns (total 18 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   CUST_ID                           8950 non-null   object 
     1   BALANCE                           8950 non-null   float64
     2   BALANCE_FREQUENCY                 8950 non-null   float64
     3   PURCHASES                         8950 non-null   float64
     4   ONEOFF_PURCHASES                  8950 non-null   float64
     5   INSTALLMENTS_PURCHASES            8950 non-null   float64
     6   CASH_ADVANCE                      8950 non-null   float64
     7   PURCHASES_FREQUENCY               8950 non-null   float64
     8   ONEOFF_PURCHASES_FREQUENCY        8950 non-null   float64
     9   PURCHASES_INSTALLMENTS_FREQUENCY  8950 non-null   float64
     10  CASH_ADVANCE_FREQUENCY            8950 non-null   float64
     11  CASH_ADVANCE_TRX                  8950 non-null   int64  
     12  PURCHASES_TRX                     8950 non-null   int64  
     13  CREDIT_LIMIT                      8949 non-null   float64
     14  PAYMENTS                          8950 non-null   float64
     15  MINIMUM_PAYMENTS                  8637 non-null   float64
     16  PRC_FULL_PAYMENT                  8950 non-null   float64
     17  TENURE                            8950 non-null   int64  
    dtypes: float64(14), int64(3), object(1)
    memory usage: 1.2+ MB
    

There are 18 columns in this dataset. The CUST_ID is an object and it is the customer ID that is used to identify the customer. We may drop it since it is not one of the behavior features.  CASH_ADVANCE_TRX, PURCHASES_TRX, and TENURE are integers. Any other columns are float data types. 
<ol>
    <li>CUST_ID: ID of Credit Card holder</li>
    <li>BALANCE: Amount left in their account to make purchases</li>
    <li>BALANCE_FREQUENCY: The frequency of the balance is updated, the score is between 0 and 1 (1 = frequently updated, 0 = not frequently updated)</li>
    <li>PURCHASES: Amount of purchases made from the account</li>
    <li>ONEOFF_PURCHASES: Maximum purchase amount done in one attempt</li>
    <li>INSTALLMENTS_PURCHASES: Amount of purchase done in installment</li>
    <li>CASH_ADVANCE: Cash in advance given by the user. A cash advance is a service provided by credit card issuers that allows cardholders to immediately withdraw a sum of cash, often at a high interest rate.</li>
    <li>PURCHASES_FREQUENCY: The frequency of the purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased) </li>
    <li>ONEOFF_PURCHASES_FREQUENCY: The frequency of the purchases done in one attempt (1 = frequently purchased, 0 = not frequently purchased)</li>
    <li>PURCHASES_INSTALLMENTS_FREQUENCY: the frequency of the purchases in installments are being done (1 = frequently done, 0 = not frequently done)</li>
    <li>CASH_ADVANCE_FREQUENCY: The frequency of the cash in advance given by the user</li>
    <li>CASH_ADVANCE_TRX: Number of cash advance transactions being made</li>
    <li>PURCHASES_TRX: Number of purchase transactions being made</li>
    <li>CREDIT_LIMIT: The limit of credit card for user</li>
    <li>PAYMENTS: Amount of payment done by user</li>
    <li>MINIMUM_PAYMENTS: Minimum amount of payments made by the user</li>
    <li>PRC_FULL_PAYMENT: Percent of full payment paid by the user, score between 0 and 1</li>
    <li>TENURE: Tenure of credit card service for user</li>
</ol>

### Data Cleaning



```python
df.isnull().values.any()
```




    True




```python
df.isnull().sum()
```




    CUST_ID                               0
    BALANCE                               0
    BALANCE_FREQUENCY                     0
    PURCHASES                             0
    ONEOFF_PURCHASES                      0
    INSTALLMENTS_PURCHASES                0
    CASH_ADVANCE                          0
    PURCHASES_FREQUENCY                   0
    ONEOFF_PURCHASES_FREQUENCY            0
    PURCHASES_INSTALLMENTS_FREQUENCY      0
    CASH_ADVANCE_FREQUENCY                0
    CASH_ADVANCE_TRX                      0
    PURCHASES_TRX                         0
    CREDIT_LIMIT                          1
    PAYMENTS                              0
    MINIMUM_PAYMENTS                    313
    PRC_FULL_PAYMENT                      0
    TENURE                                0
    dtype: int64



There are 313 of MINIMUM_PAYMENTS and 1 CREDIT_LIMIT have null value. 


```python
df[['CREDIT_LIMIT', 'MINIMUM_PAYMENTS']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CREDIT_LIMIT</th>
      <th>MINIMUM_PAYMENTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8949.000000</td>
      <td>8637.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4494.449450</td>
      <td>864.206542</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3638.815725</td>
      <td>2372.446607</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.000000</td>
      <td>0.019163</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1600.000000</td>
      <td>169.123707</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3000.000000</td>
      <td>312.343947</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6500.000000</td>
      <td>825.485459</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30000.000000</td>
      <td>76406.207520</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['CREDIT_LIMIT'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5203</th>
      <td>C15349</td>
      <td>18.400472</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>186.853063</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>9.040017</td>
      <td>14.418723</td>
      <td>0.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Because there is only one customer with an empty CREDIT_LIMIT, we can drop this row. 


```python
df.drop(index=[df[df['CREDIT_LIMIT'].isna()].index[0]], inplace=True)
```


```python
df.shape
```




    (8949, 18)




```python
df[df['MINIMUM_PAYMENTS'].isna()].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>C10004</td>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>45</th>
      <td>C10047</td>
      <td>2242.311686</td>
      <td>1.000000</td>
      <td>437.00</td>
      <td>97.00</td>
      <td>340.0</td>
      <td>184.648692</td>
      <td>0.333333</td>
      <td>0.083333</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>2</td>
      <td>5</td>
      <td>2400.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>47</th>
      <td>C10049</td>
      <td>3910.111237</td>
      <td>1.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1980.873201</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>7</td>
      <td>0</td>
      <td>4200.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>54</th>
      <td>C10056</td>
      <td>6.660517</td>
      <td>0.636364</td>
      <td>310.00</td>
      <td>0.00</td>
      <td>310.0</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0</td>
      <td>8</td>
      <td>1000.0</td>
      <td>417.016763</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>55</th>
      <td>C10057</td>
      <td>1311.995984</td>
      <td>1.000000</td>
      <td>1283.90</td>
      <td>1283.90</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>6000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>56</th>
      <td>C10058</td>
      <td>3625.218146</td>
      <td>1.000000</td>
      <td>313.27</td>
      <td>313.27</td>
      <td>0.0</td>
      <td>668.468743</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.416667</td>
      <td>5</td>
      <td>4</td>
      <td>4000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>63</th>
      <td>C10065</td>
      <td>7.152356</td>
      <td>0.090909</td>
      <td>840.00</td>
      <td>840.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1600.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>93</th>
      <td>C10098</td>
      <td>1307.717841</td>
      <td>1.000000</td>
      <td>405.60</td>
      <td>405.60</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>2400.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>94</th>
      <td>C10099</td>
      <td>2329.485768</td>
      <td>1.000000</td>
      <td>213.34</td>
      <td>213.34</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>3</td>
      <td>2400.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>97</th>
      <td>C10102</td>
      <td>3505.671311</td>
      <td>1.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1713.984305</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>6</td>
      <td>0</td>
      <td>4000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
```


```python
df.isnull().sum()
```




    CUST_ID                             0
    BALANCE                             0
    BALANCE_FREQUENCY                   0
    PURCHASES                           0
    ONEOFF_PURCHASES                    0
    INSTALLMENTS_PURCHASES              0
    CASH_ADVANCE                        0
    PURCHASES_FREQUENCY                 0
    ONEOFF_PURCHASES_FREQUENCY          0
    PURCHASES_INSTALLMENTS_FREQUENCY    0
    CASH_ADVANCE_FREQUENCY              0
    CASH_ADVANCE_TRX                    0
    PURCHASES_TRX                       0
    CREDIT_LIMIT                        0
    PAYMENTS                            0
    MINIMUM_PAYMENTS                    0
    PRC_FULL_PAYMENT                    0
    TENURE                              0
    dtype: int64



There are no null values in the dataset. <br>We can check if there are any duplicate rows in the dataset. 


```python
df.duplicated().sum()
```




    0



There are no duplicated rows in the dataset. 

The CUST_ID is an object and it is the customer ID that is used to identify the customer. We may drop it since it is not one of the behavior features.


```python
df.drop(columns=["CUST_ID"],axis=1,inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.4</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>864.304943</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



### Analysis and Visualization

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

The describe function can help finding the min, mean, max, and standard deviation of each feature. 


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BALANCE</th>
      <td>8949.0</td>
      <td>1564.647593</td>
      <td>2081.584016</td>
      <td>0.000000</td>
      <td>128.365782</td>
      <td>873.680279</td>
      <td>2054.372848</td>
      <td>19043.13856</td>
    </tr>
    <tr>
      <th>BALANCE_FREQUENCY</th>
      <td>8949.0</td>
      <td>0.877350</td>
      <td>0.236798</td>
      <td>0.000000</td>
      <td>0.888889</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>PURCHASES</th>
      <td>8949.0</td>
      <td>1003.316936</td>
      <td>2136.727848</td>
      <td>0.000000</td>
      <td>39.800000</td>
      <td>361.490000</td>
      <td>1110.170000</td>
      <td>49039.57000</td>
    </tr>
    <tr>
      <th>ONEOFF_PURCHASES</th>
      <td>8949.0</td>
      <td>592.503572</td>
      <td>1659.968851</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>577.830000</td>
      <td>40761.25000</td>
    </tr>
    <tr>
      <th>INSTALLMENTS_PURCHASES</th>
      <td>8949.0</td>
      <td>411.113579</td>
      <td>904.378205</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>89.000000</td>
      <td>468.650000</td>
      <td>22500.00000</td>
    </tr>
    <tr>
      <th>CASH_ADVANCE</th>
      <td>8949.0</td>
      <td>978.959616</td>
      <td>2097.264344</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1113.868654</td>
      <td>47137.21176</td>
    </tr>
    <tr>
      <th>PURCHASES_FREQUENCY</th>
      <td>8949.0</td>
      <td>0.490405</td>
      <td>0.401360</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.500000</td>
      <td>0.916667</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <td>8949.0</td>
      <td>0.202480</td>
      <td>0.298345</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.300000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <td>8949.0</td>
      <td>0.364478</td>
      <td>0.397451</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.750000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <td>8949.0</td>
      <td>0.135141</td>
      <td>0.200132</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.222222</td>
      <td>1.50000</td>
    </tr>
    <tr>
      <th>CASH_ADVANCE_TRX</th>
      <td>8949.0</td>
      <td>3.249078</td>
      <td>6.824987</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>123.00000</td>
    </tr>
    <tr>
      <th>PURCHASES_TRX</th>
      <td>8949.0</td>
      <td>14.711476</td>
      <td>24.858552</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>17.000000</td>
      <td>358.00000</td>
    </tr>
    <tr>
      <th>CREDIT_LIMIT</th>
      <td>8949.0</td>
      <td>4494.449450</td>
      <td>3638.815725</td>
      <td>50.000000</td>
      <td>1600.000000</td>
      <td>3000.000000</td>
      <td>6500.000000</td>
      <td>30000.00000</td>
    </tr>
    <tr>
      <th>PAYMENTS</th>
      <td>8949.0</td>
      <td>1733.336511</td>
      <td>2895.168146</td>
      <td>0.000000</td>
      <td>383.282850</td>
      <td>857.062706</td>
      <td>1901.279320</td>
      <td>50721.48336</td>
    </tr>
    <tr>
      <th>MINIMUM_PAYMENTS</th>
      <td>8949.0</td>
      <td>864.304943</td>
      <td>2330.700932</td>
      <td>0.019163</td>
      <td>170.875613</td>
      <td>335.657631</td>
      <td>864.304943</td>
      <td>76406.20752</td>
    </tr>
    <tr>
      <th>PRC_FULL_PAYMENT</th>
      <td>8949.0</td>
      <td>0.153732</td>
      <td>0.292511</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>TENURE</th>
      <td>8949.0</td>
      <td>11.517935</td>
      <td>1.337134</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.00000</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, there are some outliers when looking at the max value. Because they could contain important information about that customer so the outliers can be treated as extreme values in this case. 

The corr function can help discover the correlation coefficient between each pair of features. 


```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BALANCE</th>
      <td>1.000000</td>
      <td>0.322335</td>
      <td>0.181230</td>
      <td>0.164326</td>
      <td>0.126437</td>
      <td>0.496679</td>
      <td>-0.078054</td>
      <td>0.073114</td>
      <td>-0.063267</td>
      <td>0.449246</td>
      <td>0.385139</td>
      <td>0.154297</td>
      <td>0.531283</td>
      <td>0.322769</td>
      <td>0.394266</td>
      <td>-0.319018</td>
      <td>0.072421</td>
    </tr>
    <tr>
      <th>BALANCE_FREQUENCY</th>
      <td>0.322335</td>
      <td>1.000000</td>
      <td>0.133586</td>
      <td>0.104257</td>
      <td>0.124204</td>
      <td>0.099312</td>
      <td>0.229440</td>
      <td>0.202295</td>
      <td>0.175869</td>
      <td>0.192022</td>
      <td>0.141516</td>
      <td>0.189527</td>
      <td>0.095843</td>
      <td>0.064842</td>
      <td>0.114182</td>
      <td>-0.095308</td>
      <td>0.118566</td>
    </tr>
    <tr>
      <th>PURCHASES</th>
      <td>0.181230</td>
      <td>0.133586</td>
      <td>1.000000</td>
      <td>0.916844</td>
      <td>0.679889</td>
      <td>-0.051495</td>
      <td>0.392991</td>
      <td>0.498413</td>
      <td>0.315537</td>
      <td>-0.120137</td>
      <td>-0.067194</td>
      <td>0.689552</td>
      <td>0.356963</td>
      <td>0.603252</td>
      <td>0.093497</td>
      <td>0.180356</td>
      <td>0.086155</td>
    </tr>
    <tr>
      <th>ONEOFF_PURCHASES</th>
      <td>0.164326</td>
      <td>0.104257</td>
      <td>0.916844</td>
      <td>1.000000</td>
      <td>0.330610</td>
      <td>-0.031341</td>
      <td>0.264913</td>
      <td>0.524881</td>
      <td>0.127699</td>
      <td>-0.082622</td>
      <td>-0.046226</td>
      <td>0.545514</td>
      <td>0.319724</td>
      <td>0.567283</td>
      <td>0.048583</td>
      <td>0.132745</td>
      <td>0.064047</td>
    </tr>
    <tr>
      <th>INSTALLMENTS_PURCHASES</th>
      <td>0.126437</td>
      <td>0.124204</td>
      <td>0.679889</td>
      <td>0.330610</td>
      <td>1.000000</td>
      <td>-0.064264</td>
      <td>0.442398</td>
      <td>0.214016</td>
      <td>0.511334</td>
      <td>-0.132312</td>
      <td>-0.074017</td>
      <td>0.628097</td>
      <td>0.256499</td>
      <td>0.384066</td>
      <td>0.131671</td>
      <td>0.182548</td>
      <td>0.086016</td>
    </tr>
    <tr>
      <th>CASH_ADVANCE</th>
      <td>0.496679</td>
      <td>0.099312</td>
      <td>-0.051495</td>
      <td>-0.031341</td>
      <td>-0.064264</td>
      <td>1.000000</td>
      <td>-0.215579</td>
      <td>-0.086786</td>
      <td>-0.177118</td>
      <td>0.628535</td>
      <td>0.656493</td>
      <td>-0.075877</td>
      <td>0.303985</td>
      <td>0.453226</td>
      <td>0.139209</td>
      <td>-0.152961</td>
      <td>-0.068552</td>
    </tr>
    <tr>
      <th>PURCHASES_FREQUENCY</th>
      <td>-0.078054</td>
      <td>0.229440</td>
      <td>0.392991</td>
      <td>0.264913</td>
      <td>0.442398</td>
      <td>-0.215579</td>
      <td>1.000000</td>
      <td>0.501305</td>
      <td>0.862921</td>
      <td>-0.308483</td>
      <td>-0.203541</td>
      <td>0.568408</td>
      <td>0.119788</td>
      <td>0.103393</td>
      <td>0.002926</td>
      <td>0.305761</td>
      <td>0.061006</td>
    </tr>
    <tr>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <td>0.073114</td>
      <td>0.202295</td>
      <td>0.498413</td>
      <td>0.524881</td>
      <td>0.214016</td>
      <td>-0.086786</td>
      <td>0.501305</td>
      <td>1.000000</td>
      <td>0.142270</td>
      <td>-0.111707</td>
      <td>-0.069116</td>
      <td>0.544849</td>
      <td>0.295038</td>
      <td>0.243503</td>
      <td>-0.029992</td>
      <td>0.157497</td>
      <td>0.082234</td>
    </tr>
    <tr>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <td>-0.063267</td>
      <td>0.175869</td>
      <td>0.315537</td>
      <td>0.127699</td>
      <td>0.511334</td>
      <td>-0.177118</td>
      <td>0.862921</td>
      <td>0.142270</td>
      <td>1.000000</td>
      <td>-0.262955</td>
      <td>-0.169250</td>
      <td>0.529949</td>
      <td>0.060755</td>
      <td>0.085496</td>
      <td>0.029554</td>
      <td>0.250049</td>
      <td>0.072926</td>
    </tr>
    <tr>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <td>0.449246</td>
      <td>0.192022</td>
      <td>-0.120137</td>
      <td>-0.082622</td>
      <td>-0.132312</td>
      <td>0.628535</td>
      <td>-0.308483</td>
      <td>-0.111707</td>
      <td>-0.262955</td>
      <td>1.000000</td>
      <td>0.799573</td>
      <td>-0.131161</td>
      <td>0.132616</td>
      <td>0.183206</td>
      <td>0.097905</td>
      <td>-0.249768</td>
      <td>-0.133427</td>
    </tr>
    <tr>
      <th>CASH_ADVANCE_TRX</th>
      <td>0.385139</td>
      <td>0.141516</td>
      <td>-0.067194</td>
      <td>-0.046226</td>
      <td>-0.074017</td>
      <td>0.656493</td>
      <td>-0.203541</td>
      <td>-0.069116</td>
      <td>-0.169250</td>
      <td>0.799573</td>
      <td>1.000000</td>
      <td>-0.066180</td>
      <td>0.149700</td>
      <td>0.255262</td>
      <td>0.109173</td>
      <td>-0.169807</td>
      <td>-0.043614</td>
    </tr>
    <tr>
      <th>PURCHASES_TRX</th>
      <td>0.154297</td>
      <td>0.189527</td>
      <td>0.689552</td>
      <td>0.545514</td>
      <td>0.628097</td>
      <td>-0.075877</td>
      <td>0.568408</td>
      <td>0.544849</td>
      <td>0.529949</td>
      <td>-0.131161</td>
      <td>-0.066180</td>
      <td>1.000000</td>
      <td>0.272882</td>
      <td>0.370807</td>
      <td>0.095836</td>
      <td>0.162037</td>
      <td>0.121719</td>
    </tr>
    <tr>
      <th>CREDIT_LIMIT</th>
      <td>0.531283</td>
      <td>0.095843</td>
      <td>0.356963</td>
      <td>0.319724</td>
      <td>0.256499</td>
      <td>0.303985</td>
      <td>0.119788</td>
      <td>0.295038</td>
      <td>0.060755</td>
      <td>0.132616</td>
      <td>0.149700</td>
      <td>0.272882</td>
      <td>1.000000</td>
      <td>0.421861</td>
      <td>0.125134</td>
      <td>0.055672</td>
      <td>0.139167</td>
    </tr>
    <tr>
      <th>PAYMENTS</th>
      <td>0.322769</td>
      <td>0.064842</td>
      <td>0.603252</td>
      <td>0.567283</td>
      <td>0.384066</td>
      <td>0.453226</td>
      <td>0.103393</td>
      <td>0.243503</td>
      <td>0.085496</td>
      <td>0.183206</td>
      <td>0.255262</td>
      <td>0.370807</td>
      <td>0.421861</td>
      <td>1.000000</td>
      <td>0.125024</td>
      <td>0.112107</td>
      <td>0.105965</td>
    </tr>
    <tr>
      <th>MINIMUM_PAYMENTS</th>
      <td>0.394266</td>
      <td>0.114182</td>
      <td>0.093497</td>
      <td>0.048583</td>
      <td>0.131671</td>
      <td>0.139209</td>
      <td>0.002926</td>
      <td>-0.029992</td>
      <td>0.029554</td>
      <td>0.097905</td>
      <td>0.109173</td>
      <td>0.095836</td>
      <td>0.125134</td>
      <td>0.125024</td>
      <td>1.000000</td>
      <td>-0.139700</td>
      <td>0.057144</td>
    </tr>
    <tr>
      <th>PRC_FULL_PAYMENT</th>
      <td>-0.319018</td>
      <td>-0.095308</td>
      <td>0.180356</td>
      <td>0.132745</td>
      <td>0.182548</td>
      <td>-0.152961</td>
      <td>0.305761</td>
      <td>0.157497</td>
      <td>0.250049</td>
      <td>-0.249768</td>
      <td>-0.169807</td>
      <td>0.162037</td>
      <td>0.055672</td>
      <td>0.112107</td>
      <td>-0.139700</td>
      <td>1.000000</td>
      <td>-0.016744</td>
    </tr>
    <tr>
      <th>TENURE</th>
      <td>0.072421</td>
      <td>0.118566</td>
      <td>0.086155</td>
      <td>0.064047</td>
      <td>0.086016</td>
      <td>-0.068552</td>
      <td>0.061006</td>
      <td>0.082234</td>
      <td>0.072926</td>
      <td>-0.133427</td>
      <td>-0.043614</td>
      <td>0.121719</td>
      <td>0.139167</td>
      <td>0.105965</td>
      <td>0.057144</td>
      <td>-0.016744</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, PURCHASES and ONEOFF_PURCHASES have a strong correlation because the magnitude is 0.916844, which is high. PURCHASES_INSTALLMENTS_FREQUENCY and PURCHASES_FREQUENCY also have a high correlation with a 0.862921. We can use the heatmap from the Seaborn library to have a better view of the correlation coefficient. 


```python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True)
plt.show()
```


    
![png](output_40_0.png)
    



```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x22ac8233a88>




    
![png](output_41_1.png)
    


Notice that some areas from the plot above are high-density. It looks like we can apply an algorithm to separate high density with a cluster of low density. 


```python
df.hist(bins=12, figsize=(20, 15), layout=(5,4));
```


    
![png](output_43_0.png)
    


From the above plots, notices that most of the graphs are skewed. The reason could be most customers have some common in one feature. 


```python
sns.scatterplot(x='PURCHASES', y='ONEOFF_PURCHASES', data=df);
```


    
![png](output_45_0.png)
    



```python
df_purchases = df[['PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']]
df_purchases.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['PURCHASES'] == df['ONEOFF_PURCHASES'] + df['INSTALLMENTS_PURCHASES']
```




    0       True
    1       True
    2       True
    3       True
    4       True
            ... 
    8945    True
    8946    True
    8947    True
    8948    True
    8949    True
    Length: 8949, dtype: bool




```python
df_purchases['SUM_OF_ONEOFF_INSTALLMENTS'] = df_purchases['ONEOFF_PURCHASES'] + df_purchases['INSTALLMENTS_PURCHASES']
df_purchases.loc[df['PURCHASES'] != df_purchases['ONEOFF_PURCHASES'] + df_purchases['INSTALLMENTS_PURCHASES']]
```

    C:\Users\jacks\anaconda3\envs\UL\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>SUM_OF_ONEOFF_INSTALLMENTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>901.42</td>
      <td>646.07</td>
      <td>255.35</td>
      <td>901.42</td>
    </tr>
    <tr>
      <th>71</th>
      <td>4523.27</td>
      <td>1664.09</td>
      <td>2859.18</td>
      <td>4523.27</td>
    </tr>
    <tr>
      <th>82</th>
      <td>133.05</td>
      <td>28.20</td>
      <td>104.85</td>
      <td>133.05</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1603.78</td>
      <td>1445.14</td>
      <td>158.64</td>
      <td>1603.78</td>
    </tr>
    <tr>
      <th>110</th>
      <td>1354.86</td>
      <td>585.63</td>
      <td>769.23</td>
      <td>1354.86</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8769</th>
      <td>1045.14</td>
      <td>227.14</td>
      <td>818.00</td>
      <td>1045.14</td>
    </tr>
    <tr>
      <th>8825</th>
      <td>654.84</td>
      <td>460.89</td>
      <td>193.95</td>
      <td>654.84</td>
    </tr>
    <tr>
      <th>8832</th>
      <td>63.40</td>
      <td>35.09</td>
      <td>28.31</td>
      <td>63.40</td>
    </tr>
    <tr>
      <th>8834</th>
      <td>510.00</td>
      <td>0.00</td>
      <td>780.00</td>
      <td>780.00</td>
    </tr>
    <tr>
      <th>8927</th>
      <td>315.20</td>
      <td>147.80</td>
      <td>167.40</td>
      <td>315.20</td>
    </tr>
  </tbody>
</table>
<p>492 rows × 4 columns</p>
</div>



From the above analysis, we can see that most of the purchase is equal to the sum of the one-off purchase and installment purchase. Only a few customers such as the one on row 8834 who has a high installment purchase. 


```python
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(df['TENURE'].value_counts(), autopct='%1.1f%%', pctdistance=1.1)
ax1.legend(df['TENURE'].value_counts().index)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Percentage by the Tenure")
plt.show()
```


    
![png](output_50_0.png)
    


From the pie chart above, we can see about 84.7% of users have a 12 months tenure. 


```python
sns.boxplot(x = 'TENURE', y = 'CREDIT_LIMIT', data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae07137c8>




    
![png](output_52_1.png)
    


From the boxplots above, we can see that user who have longer tenure also tends to have a higher credit limit. 


```python

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(df['PRC_FULL_PAYMENT'].value_counts(), autopct='%1.1f%%', pctdistance=1.1)
ax1.legend(df['PRC_FULL_PAYMENT'].value_counts().index)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Percentage by the PRC_FULL_PAYMENT")
plt.show()
```


    
![png](output_54_0.png)
    


From the pie chart above, only 5.5% of users made a full payment. Surprisingly, about 66% of users with 0% of full payment paid. Users who made a full payment could have enough money in their savings, company may offer a wealth management plan or saving plan to those users. 


```python
sns.scatterplot(x='BALANCE', y='PURCHASES', data=df);
```


    
![png](output_56_0.png)
    


It makes sense when the amount of purchases made is below the balance amount left in their account. There are some outliers such as the user who only has a balance of about \\$11,000 but has \\$50,000 purchases. Those users could be business owners who may need a large amount of money so they may need a loan to purchase more. 


```python
sns.scatterplot(x='CASH_ADVANCE', y='CASH_ADVANCE_TRX', data=df);
```


    
![png](output_58_0.png)
    



```python
sns.scatterplot(x='CASH_ADVANCE', y='PAYMENTS',data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae0ce05c8>




    
![png](output_59_1.png)
    



```python
sns.scatterplot(x='CASH_ADVANCE_TRX', y='PAYMENTS',data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae0cb5b08>




    
![png](output_60_1.png)
    


Cash Advance is like a short-term loan offered by credit card issuers. People who use cash advance a lot is more likely to need a loan. The user who likes taking cash advances but only makes a small amount of payments could be a customer who likes to borrow a loan but may have issues paying off the loan in the future. 


```python
sns.scatterplot(x='CASH_ADVANCE', y='BALANCE',data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae0d9dc08>




    
![png](output_62_1.png)
    


People who have a high balance and high cash advance have a high probability to apply for a loan. 


```python
sns.scatterplot(x='CREDIT_LIMIT', y='PURCHASES',data = df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae0e56048>




    
![png](output_64_1.png)
    


There is a small group of users who make purchases higher than the credit limit, they could be the customer who needs a loan but users with a low credit limit could have a bad credit history. 

### Preprocessing

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

Before applying the data to the unsupervised model, we need to **standardize** the data. Data standardization transform features to a **similar scale**. It rescales data to have a mean of 0 and a standard deviation of 1. From the analysis above, we can see that some features are from 0 to 1 but some features have a wide range of scope. The dataset has extremely high or low values. Standardization can transform the dataset to a common scale so the training won't affect by the large different ranges of values.


```python
scaler = StandardScaler()
data=scaler.fit_transform(df)
data = pd.DataFrame(data, columns=df.columns)
```

Let's see what the data looks like after standarization:


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.732054</td>
      <td>-0.249881</td>
      <td>-0.424934</td>
      <td>-0.356957</td>
      <td>-0.349114</td>
      <td>-0.466805</td>
      <td>-0.806649</td>
      <td>-0.678716</td>
      <td>-0.707409</td>
      <td>-0.675294</td>
      <td>-0.476083</td>
      <td>-0.511381</td>
      <td>-0.960380</td>
      <td>-0.529026</td>
      <td>-3.109947e-01</td>
      <td>-0.525588</td>
      <td>0.360541</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.786858</td>
      <td>0.134049</td>
      <td>-0.469584</td>
      <td>-0.356957</td>
      <td>-0.454607</td>
      <td>2.605438</td>
      <td>-1.221928</td>
      <td>-0.678716</td>
      <td>-0.917090</td>
      <td>0.573949</td>
      <td>0.110032</td>
      <td>-0.591841</td>
      <td>0.688601</td>
      <td>0.818546</td>
      <td>8.926366e-02</td>
      <td>0.234159</td>
      <td>0.360541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.447041</td>
      <td>0.517980</td>
      <td>-0.107716</td>
      <td>0.108843</td>
      <td>-0.454607</td>
      <td>-0.466805</td>
      <td>1.269742</td>
      <td>2.673295</td>
      <td>-0.917090</td>
      <td>-0.675294</td>
      <td>-0.476083</td>
      <td>-0.109082</td>
      <td>0.826016</td>
      <td>-0.383857</td>
      <td>-1.017005e-01</td>
      <td>-0.525588</td>
      <td>0.360541</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.049015</td>
      <td>-1.017743</td>
      <td>0.231995</td>
      <td>0.546123</td>
      <td>-0.454607</td>
      <td>-0.368678</td>
      <td>-1.014290</td>
      <td>-0.399383</td>
      <td>-0.917090</td>
      <td>-0.258882</td>
      <td>-0.329554</td>
      <td>-0.551611</td>
      <td>0.826016</td>
      <td>-0.598733</td>
      <td>4.878069e-17</td>
      <td>-0.525588</td>
      <td>0.360541</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.358849</td>
      <td>0.517980</td>
      <td>-0.462095</td>
      <td>-0.347317</td>
      <td>-0.454607</td>
      <td>-0.466805</td>
      <td>-1.014290</td>
      <td>-0.399383</td>
      <td>-0.917090</td>
      <td>-0.675294</td>
      <td>-0.476083</td>
      <td>-0.551611</td>
      <td>-0.905414</td>
      <td>-0.364421</td>
      <td>-2.658206e-01</td>
      <td>-0.525588</td>
      <td>0.360541</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8949.000000</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
      <td>8.949000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000000</td>
      <td>-3.144205e-16</td>
      <td>-8.257509e-17</td>
      <td>-4.128754e-17</td>
      <td>1.746781e-17</td>
      <td>-1.587982e-17</td>
      <td>7.701715e-17</td>
      <td>-6.351930e-18</td>
      <td>4.605149e-17</td>
      <td>-1.270386e-17</td>
      <td>1.587982e-17</td>
      <td>1.905579e-17</td>
      <td>2.159656e-16</td>
      <td>6.351930e-17</td>
      <td>1.270386e-17</td>
      <td>-3.175965e-18</td>
      <td>-2.794849e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000056</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
      <td>1.000056e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.751704</td>
      <td>-3.705263e+00</td>
      <td>-4.695839e-01</td>
      <td>-3.569565e-01</td>
      <td>-4.546069e-01</td>
      <td>-4.668054e-01</td>
      <td>-1.221928e+00</td>
      <td>-6.787162e-01</td>
      <td>-9.170895e-01</td>
      <td>-6.752945e-01</td>
      <td>-4.760829e-01</td>
      <td>-5.918405e-01</td>
      <td>-1.221468e+00</td>
      <td>-5.987332e-01</td>
      <td>-3.708473e-01</td>
      <td>-5.255884e-01</td>
      <td>-4.126919e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.690033</td>
      <td>4.873144e-02</td>
      <td>-4.509562e-01</td>
      <td>-3.569565e-01</td>
      <td>-4.546069e-01</td>
      <td>-4.668054e-01</td>
      <td>-1.014290e+00</td>
      <td>-6.787162e-01</td>
      <td>-9.170895e-01</td>
      <td>-6.752945e-01</td>
      <td>-4.760829e-01</td>
      <td>-5.516107e-01</td>
      <td>-7.954817e-01</td>
      <td>-4.663388e-01</td>
      <td>-2.975363e-01</td>
      <td>-5.255884e-01</td>
      <td>3.605413e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.331962</td>
      <td>5.179802e-01</td>
      <td>-3.003952e-01</td>
      <td>-3.340632e-01</td>
      <td>-3.561912e-01</td>
      <td>-4.668054e-01</td>
      <td>2.390672e-02</td>
      <td>-3.993831e-01</td>
      <td>-4.977267e-01</td>
      <td>-6.752945e-01</td>
      <td>-4.760829e-01</td>
      <td>-3.102316e-01</td>
      <td>-4.107196e-01</td>
      <td>-3.026846e-01</td>
      <td>-2.268317e-01</td>
      <td>-5.255884e-01</td>
      <td>3.605413e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.235279</td>
      <td>5.179802e-01</td>
      <td>5.001060e-02</td>
      <td>-8.840161e-03</td>
      <td>6.362342e-02</td>
      <td>6.432979e-02</td>
      <td>1.062103e+00</td>
      <td>3.268871e-01</td>
      <td>9.700395e-01</td>
      <td>4.351431e-01</td>
      <td>1.100315e-01</td>
      <td>9.206698e-02</td>
      <td>5.511855e-01</td>
      <td>5.801121e-02</td>
      <td>4.878069e-17</td>
      <td>-3.717957e-02</td>
      <td>3.605413e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.397195</td>
      <td>5.179802e-01</td>
      <td>2.248248e+01</td>
      <td>2.419985e+01</td>
      <td>2.442576e+01</td>
      <td>2.201002e+01</td>
      <td>1.269742e+00</td>
      <td>2.673295e+00</td>
      <td>1.599083e+00</td>
      <td>6.820167e+00</td>
      <td>1.754694e+01</td>
      <td>1.381045e+01</td>
      <td>7.009692e+00</td>
      <td>1.692160e+01</td>
      <td>3.241348e+01</td>
      <td>2.893277e+00</td>
      <td>3.605413e-01</td>
    </tr>
  </tbody>
</table>
</div>



### Models

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

#### K-means Clustering

**K-means clustering** is one of the most popular techniques in unsupervised machine learning, which searches k clusters in your data. <br>

Main steps:<br>
<ol>
    <li>pick k (number of clusters)</li>
    <li>place k centroids randomly among the training data</li>
    <li>calculate distance from each centroid to all the points in the training data</li>
    <li>group all the data points with their nearest centroid</li>
    <li>calculate the mean data point in a single cluster and move the previous centroid to the mean location</li>
    <li>repeat for each cluster</li>
    <li>repeat the step 2 to step 6 until centroids don't move and colors don't change or maximum number of iterations has been achieved</li>
</ol> 

Let's start with a 3 clusters model.


```python
km_3 = KMeans(3)
km_3_clusters = km_3.fit_predict(data)
```

**Silhouette score** can help evaluate the performance of unsupervised learning methods.The silhouette score is a metric that evaluates how well the data points fit in their clusters. Simplified Silhouette Index = (bi-ai)/(max(ai, bi)), where ai is the distance from data point i to its own cluster centroid and bi is the distance from point i to the nearest cluster centroid. The score ranges from -1 to 1, where 1 indicates the model achieved perfect clusters. 


```python
silhouette_score(data, km_3_clusters)
```




    0.2511201158410639



Let's see what it looks like with some plots.


```python
sns.scatterplot(x='CREDIT_LIMIT', y='PURCHASES',data = data,hue = km_3_clusters)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae0edd8c8>




    
![png](output_79_1.png)
    


From the plot above, we can see cluster one are customer with a higher purchase. Cluster zero and cluster two are mixing together at the bottom. 


```python
sns.scatterplot(x='PURCHASES', y='ONEOFF_PURCHASES', data=data,hue = km_3_clusters)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae11c5a48>




    
![png](output_81_1.png)
    


It is a little bit hard to see the clusters with these features. 

Let's build a for loop for different k values. 


```python
km_list = []
for i in range (2,11):
    km = KMeans(i)
    km_clusters = km.fit_predict(data)
    sil_score = silhouette_score(data, km_clusters)
    print(f"k={i} K-Means Clustering: {sil_score}")
    
    km_list.append((i, sil_score))
    
    plt.scatter(x='CREDIT_LIMIT', y='PURCHASES',data = data,c = km_clusters)
    plt.title(f"Distribution of K-means clusters based on Credit limit and total purchases when k={i}")
    plt.show()

df_km = pd.DataFrame(km_list, columns=['k', 'silhouette_score'])
```

    k=2 K-Means Clustering: 0.20948941929133194
    


    
![png](output_84_1.png)
    


    k=3 K-Means Clustering: 0.2502389638054194
    


    
![png](output_84_3.png)
    


    k=4 K-Means Clustering: 0.19759862081695156
    


    
![png](output_84_5.png)
    


    k=5 K-Means Clustering: 0.19316716374436665
    


    
![png](output_84_7.png)
    


    k=6 K-Means Clustering: 0.2025496307659155
    


    
![png](output_84_9.png)
    


    k=7 K-Means Clustering: 0.21418199158286824
    


    
![png](output_84_11.png)
    


    k=8 K-Means Clustering: 0.2222800137075136
    


    
![png](output_84_13.png)
    


    k=9 K-Means Clustering: 0.2127706530371557
    


    
![png](output_84_15.png)
    


    k=10 K-Means Clustering: 0.22131963987733896
    


    
![png](output_84_17.png)
    



```python
df_km.sort_values('silhouette_score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.250239</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.222280</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>0.221320</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.214182</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>0.212771</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.209489</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.202550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.197599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.193167</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, k = 3 has the highest silhouette score. 

#### Hierarchical Clustering

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

**Agglomerative hierarchical clustering** treats each data point as its own cluster and then merges similar points together. **Linkage** defines the calculation of the distances between clusters. 

Main steps:<br>
<ol>
    <li>given n data points, treat each point as an individual cluster</li>
    <li>calculate distance between the centroids of all the clusters in the data</li>
    <li>group the closest clusters or points</li>
    <li>repeat step 2 and step 3 until there is only one single cluster</li>
    <li>plot a dendrogram(tree plots)</li>
</ol>


```python
ac = AgglomerativeClustering(linkage='average')
ac_clusters = ac.fit_predict(data)
silhouette_score(data, ac_clusters)
```




    0.8496907145224083



The silhouette_score of 0.8497 is higher. 


```python
sns.scatterplot(x='CREDIT_LIMIT', y='PURCHASES',data = data,hue = ac_clusters)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae3625048>




    
![png](output_91_1.png)
    


It looks like it group customer based on the purchase amount but only a few points label as cluster 1. 


```python
ac = AgglomerativeClustering(linkage='ward')
ac_clusters = ac.fit_predict(data)
silhouette_score(data, ac_clusters)
```




    0.18946426808640232




```python
sns.scatterplot(x='CREDIT_LIMIT', y='PURCHASES',data = data,hue = ac_clusters)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22ae3690088>




    
![png](output_94_1.png)
    


The silhouette score of ward method is low; however, it creates a different clusters with more number of point with a lable of cluster 1. 

Let's build a for loop trying with different number of clusters and different linkage methods.


```python
ac_list = []

for i in range (2,11):
    for linkage_method in ['single', 'ward', 'average', 'complete']:
        ac = AgglomerativeClustering(n_clusters=i, linkage=linkage_method)
        ac_clusters = ac.fit_predict(data)
        sil_score = silhouette_score(data, ac_clusters)
        print(f"n_clusters={i}, linkage={linkage_method}   Agglomerative Clustering: {sil_score}")
        
        ac_list.append((i, linkage_method, sil_score))

        plt.scatter(x='CREDIT_LIMIT', y='PURCHASES',data = data,c = ac_clusters)
        plt.title(f"Distribution of Agglomerative clusters (n_clusters={i}, linkage={linkage_method}) based on Credit Limit and Purchases")
        plt.show()
```

    n_clusters=2, linkage=single   Agglomerative Clustering: 0.8408048261410714
    


    
![png](output_97_1.png)
    


    n_clusters=2, linkage=ward   Agglomerative Clustering: 0.18946426808640232
    


    
![png](output_97_3.png)
    


    n_clusters=2, linkage=average   Agglomerative Clustering: 0.8496907145224083
    


    
![png](output_97_5.png)
    


    n_clusters=2, linkage=complete   Agglomerative Clustering: 0.7865446362943753
    


    
![png](output_97_7.png)
    


    n_clusters=3, linkage=single   Agglomerative Clustering: 0.8379826851553768
    


    
![png](output_97_9.png)
    


    n_clusters=3, linkage=ward   Agglomerative Clustering: 0.1816063879990238
    


    
![png](output_97_11.png)
    


    n_clusters=3, linkage=average   Agglomerative Clustering: 0.8391200448529502
    


    
![png](output_97_13.png)
    


    n_clusters=3, linkage=complete   Agglomerative Clustering: 0.7862981537029926
    


    
![png](output_97_15.png)
    


    n_clusters=4, linkage=single   Agglomerative Clustering: 0.8121063486405521
    


    
![png](output_97_17.png)
    


    n_clusters=4, linkage=ward   Agglomerative Clustering: 0.18238090388475459
    


    
![png](output_97_19.png)
    


    n_clusters=4, linkage=average   Agglomerative Clustering: 0.8129521420619767
    


    
![png](output_97_21.png)
    


    n_clusters=4, linkage=complete   Agglomerative Clustering: 0.7792493597154928
    


    
![png](output_97_23.png)
    


    n_clusters=5, linkage=single   Agglomerative Clustering: 0.8116260348700032
    


    
![png](output_97_25.png)
    


    n_clusters=5, linkage=ward   Agglomerative Clustering: 0.1570623296106423
    


    
![png](output_97_27.png)
    


    n_clusters=5, linkage=average   Agglomerative Clustering: 0.8084580367296601
    


    
![png](output_97_29.png)
    


    n_clusters=5, linkage=complete   Agglomerative Clustering: 0.6599101523037232
    


    
![png](output_97_31.png)
    


    n_clusters=6, linkage=single   Agglomerative Clustering: 0.7703226236486114
    


    
![png](output_97_33.png)
    


    n_clusters=6, linkage=ward   Agglomerative Clustering: 0.14181282927536704
    


    
![png](output_97_35.png)
    


    n_clusters=6, linkage=average   Agglomerative Clustering: 0.7438494152086946
    


    
![png](output_97_37.png)
    


    n_clusters=6, linkage=complete   Agglomerative Clustering: 0.6597986147910581
    


    
![png](output_97_39.png)
    


    n_clusters=7, linkage=single   Agglomerative Clustering: 0.770464269594815
    


    
![png](output_97_41.png)
    


    n_clusters=7, linkage=ward   Agglomerative Clustering: 0.16042534653321708
    


    
![png](output_97_43.png)
    


    n_clusters=7, linkage=average   Agglomerative Clustering: 0.743695445530731
    


    
![png](output_97_45.png)
    


    n_clusters=7, linkage=complete   Agglomerative Clustering: 0.6569799824431399
    


    
![png](output_97_47.png)
    


    n_clusters=8, linkage=single   Agglomerative Clustering: 0.7701091144948128
    


    
![png](output_97_49.png)
    


    n_clusters=8, linkage=ward   Agglomerative Clustering: 0.16170524524009727
    


    
![png](output_97_51.png)
    


    n_clusters=8, linkage=average   Agglomerative Clustering: 0.7180046252630964
    


    
![png](output_97_53.png)
    


    n_clusters=8, linkage=complete   Agglomerative Clustering: 0.6458092042734301
    


    
![png](output_97_55.png)
    


    n_clusters=9, linkage=single   Agglomerative Clustering: 0.7703512894586495
    


    
![png](output_97_57.png)
    


    n_clusters=9, linkage=ward   Agglomerative Clustering: 0.1648934453693508
    


    
![png](output_97_59.png)
    


    n_clusters=9, linkage=average   Agglomerative Clustering: 0.7023015649732002
    


    
![png](output_97_61.png)
    


    n_clusters=9, linkage=complete   Agglomerative Clustering: 0.5030745085813269
    


    
![png](output_97_63.png)
    


    n_clusters=10, linkage=single   Agglomerative Clustering: 0.7514474288644329
    


    
![png](output_97_65.png)
    


    n_clusters=10, linkage=ward   Agglomerative Clustering: 0.16692428136909315
    


    
![png](output_97_67.png)
    


    n_clusters=10, linkage=average   Agglomerative Clustering: 0.6673091161492336
    


    
![png](output_97_69.png)
    


    n_clusters=10, linkage=complete   Agglomerative Clustering: 0.5030500163654514
    


    
![png](output_97_71.png)
    



```python
df_ac = pd.DataFrame(ac_list, columns=['number_of_clusters', 'linkage_method', 'silhouette_score'])
```


```python
df_ac.sort_values('silhouette_score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_clusters</th>
      <th>linkage_method</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>average</td>
      <td>0.849691</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>single</td>
      <td>0.840805</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>average</td>
      <td>0.839120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>single</td>
      <td>0.837983</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>average</td>
      <td>0.812952</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>single</td>
      <td>0.812106</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>single</td>
      <td>0.811626</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>average</td>
      <td>0.808458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>complete</td>
      <td>0.786545</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>complete</td>
      <td>0.786298</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>complete</td>
      <td>0.779249</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>single</td>
      <td>0.770464</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9</td>
      <td>single</td>
      <td>0.770351</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>single</td>
      <td>0.770323</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8</td>
      <td>single</td>
      <td>0.770109</td>
    </tr>
    <tr>
      <th>32</th>
      <td>10</td>
      <td>single</td>
      <td>0.751447</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6</td>
      <td>average</td>
      <td>0.743849</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>average</td>
      <td>0.743695</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8</td>
      <td>average</td>
      <td>0.718005</td>
    </tr>
    <tr>
      <th>30</th>
      <td>9</td>
      <td>average</td>
      <td>0.702302</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10</td>
      <td>average</td>
      <td>0.667309</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>complete</td>
      <td>0.659910</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>complete</td>
      <td>0.659799</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>complete</td>
      <td>0.656980</td>
    </tr>
    <tr>
      <th>27</th>
      <td>8</td>
      <td>complete</td>
      <td>0.645809</td>
    </tr>
    <tr>
      <th>31</th>
      <td>9</td>
      <td>complete</td>
      <td>0.503075</td>
    </tr>
    <tr>
      <th>35</th>
      <td>10</td>
      <td>complete</td>
      <td>0.503050</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ward</td>
      <td>0.189464</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>ward</td>
      <td>0.182381</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>ward</td>
      <td>0.181606</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10</td>
      <td>ward</td>
      <td>0.166924</td>
    </tr>
    <tr>
      <th>29</th>
      <td>9</td>
      <td>ward</td>
      <td>0.164893</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8</td>
      <td>ward</td>
      <td>0.161705</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>ward</td>
      <td>0.160425</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>ward</td>
      <td>0.157062</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>ward</td>
      <td>0.141813</td>
    </tr>
  </tbody>
</table>
</div>



From the above table, the single method have generated high silhouette score; however, the plots show that it only classify a few points for a single cluster. The top 8 silhouette score both have the issues that only classify a small number of points for a single cluster, which is not good. The **complete method with n_clusters equal to 2** seems to have a well classify plot with a high silhouette score of 0.7865.

Let's see what the Dendrogram looks like for complete method. 


```python
plt.figure(figsize=(12, 5))
dendrogram = sch.dendrogram(sch.linkage(data, method = 'complete'))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](output_102_0.png)
    


#### Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

**DBSCAN** groups together data points that are close to each other based on distance measurement and minimum points. The **eps** parameter controls the maximum distance between two points. The **min_samples** parameter sets the number of points in a neighbourhood for a data point to be considered as a core point. 

Set min_samples as the number of features times two. 


```python
min_samples = data.shape[1] * 2
min_samples
```




    34



Using knn to find the eps value.


```python
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(data)
distances, indices = neighbors_fit.kneighbors(data)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
```




    [<matplotlib.lines.Line2D at 0x22aea0e6ec8>]




    
![png](output_108_1.png)
    


From the plot above, the "elbow" optimization point is around 2; therefore, the optimal value for eps could be around 2. 


```python
db = DBSCAN(eps=2, min_samples=min_samples, metric='euclidean')
db_clusters = db.fit_predict(data)
silhouette_score(data, db_clusters)
```




    0.41266254910755135



Evaluate DBSCAN hyperparameters on silhouette score and plots.


```python
db_list = []

#Evaluate DBSCAN hyperparameters and their effect on the silhouette score
for ep in np.arange(1, 3, 0.5):
    for min_sample in range(10, 40, 4):
        db = DBSCAN(eps=ep, min_samples = min_sample)
        db_clusters = db.fit_predict(data)
        sil_score = silhouette_score(data, db_clusters)
        db_list.append((ep, min_sample, sil_score, len(set(db.labels_))))

        plt.scatter(x='CREDIT_LIMIT', y='PURCHASES',data = data,c = db_clusters)
        plt.title('Epsilon: ' + str(ep) + ' | Minimum Points: ' + str(min_sample))
        plt.show()

        print("Silhouette Score: ", sil_score)
```


    
![png](output_112_0.png)
    


    Silhouette Score:  0.08504497694210986
    


    
![png](output_112_2.png)
    


    Silhouette Score:  -0.013882014663629535
    


    
![png](output_112_4.png)
    


    Silhouette Score:  0.042634762078436333
    


    
![png](output_112_6.png)
    


    Silhouette Score:  0.012442624140333033
    


    
![png](output_112_8.png)
    


    Silhouette Score:  -0.03007406704183043
    


    
![png](output_112_10.png)
    


    Silhouette Score:  0.054439521839534266
    


    
![png](output_112_12.png)
    


    Silhouette Score:  0.05012566025146029
    


    
![png](output_112_14.png)
    


    Silhouette Score:  0.0461804893935456
    


    
![png](output_112_16.png)
    


    Silhouette Score:  0.33618711608154717
    


    
![png](output_112_18.png)
    


    Silhouette Score:  0.3216869587664936
    


    
![png](output_112_20.png)
    


    Silhouette Score:  0.3123072446150116
    


    
![png](output_112_22.png)
    


    Silhouette Score:  0.3039429275509041
    


    
![png](output_112_24.png)
    


    Silhouette Score:  0.2975929036630351
    


    
![png](output_112_26.png)
    


    Silhouette Score:  0.29066723851517684
    


    
![png](output_112_28.png)
    


    Silhouette Score:  0.28216753530099
    


    
![png](output_112_30.png)
    


    Silhouette Score:  0.16999004397116713
    


    
![png](output_112_32.png)
    


    Silhouette Score:  0.46428871096845464
    


    
![png](output_112_34.png)
    


    Silhouette Score:  0.4529243567463877
    


    
![png](output_112_36.png)
    


    Silhouette Score:  0.4434604403205155
    


    
![png](output_112_38.png)
    


    Silhouette Score:  0.4330403686036726
    


    
![png](output_112_40.png)
    


    Silhouette Score:  0.42561368025633634
    


    
![png](output_112_42.png)
    


    Silhouette Score:  0.41831003058623617
    


    
![png](output_112_44.png)
    


    Silhouette Score:  0.41266254910755135
    


    
![png](output_112_46.png)
    


    Silhouette Score:  0.40798010436425314
    


    
![png](output_112_48.png)
    


    Silhouette Score:  0.5540906162134348
    


    
![png](output_112_50.png)
    


    Silhouette Score:  0.5480030549991355
    


    
![png](output_112_52.png)
    


    Silhouette Score:  0.5418810282513913
    


    
![png](output_112_54.png)
    


    Silhouette Score:  0.5327052825006394
    


    
![png](output_112_56.png)
    


    Silhouette Score:  0.5282829262733281
    


    
![png](output_112_58.png)
    


    Silhouette Score:  0.5226983815563176
    


    
![png](output_112_60.png)
    


    Silhouette Score:  0.5164094952690357
    


    
![png](output_112_62.png)
    


    Silhouette Score:  0.5130955696036055
    


```python
df_db = pd.DataFrame(db_list, columns=['Epsilon', 'Minimum Sample', 'Silhouette Score', 'Number of clusters'])
df_db
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epsilon</th>
      <th>Minimum Sample</th>
      <th>Silhouette Score</th>
      <th>Number of clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>10</td>
      <td>0.085045</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>14</td>
      <td>-0.013882</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>18</td>
      <td>0.042635</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>22</td>
      <td>0.012443</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>26</td>
      <td>-0.030074</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>30</td>
      <td>0.054440</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>34</td>
      <td>0.050126</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>38</td>
      <td>0.046180</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.5</td>
      <td>10</td>
      <td>0.336187</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.5</td>
      <td>14</td>
      <td>0.321687</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.5</td>
      <td>18</td>
      <td>0.312307</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.5</td>
      <td>22</td>
      <td>0.303943</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.5</td>
      <td>26</td>
      <td>0.297593</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.5</td>
      <td>30</td>
      <td>0.290667</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.5</td>
      <td>34</td>
      <td>0.282168</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.5</td>
      <td>38</td>
      <td>0.169990</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.0</td>
      <td>10</td>
      <td>0.464289</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.0</td>
      <td>14</td>
      <td>0.452924</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.0</td>
      <td>18</td>
      <td>0.443460</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.0</td>
      <td>22</td>
      <td>0.433040</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.0</td>
      <td>26</td>
      <td>0.425614</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2.0</td>
      <td>30</td>
      <td>0.418310</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2.0</td>
      <td>34</td>
      <td>0.412663</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.0</td>
      <td>38</td>
      <td>0.407980</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2.5</td>
      <td>10</td>
      <td>0.554091</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.5</td>
      <td>14</td>
      <td>0.548003</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.5</td>
      <td>18</td>
      <td>0.541881</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2.5</td>
      <td>22</td>
      <td>0.532705</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2.5</td>
      <td>26</td>
      <td>0.528283</td>
      <td>2</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2.5</td>
      <td>30</td>
      <td>0.522698</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2.5</td>
      <td>34</td>
      <td>0.516409</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2.5</td>
      <td>38</td>
      <td>0.513096</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_db.sort_values(by=['Silhouette Score'], ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epsilon</th>
      <th>Minimum Sample</th>
      <th>Silhouette Score</th>
      <th>Number of clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>2.5</td>
      <td>10</td>
      <td>0.554091</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.5</td>
      <td>14</td>
      <td>0.548003</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.5</td>
      <td>18</td>
      <td>0.541881</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2.5</td>
      <td>22</td>
      <td>0.532705</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2.5</td>
      <td>26</td>
      <td>0.528283</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



The best performance is the model with eps=2.5 and min_samples=10. The model classify the data points into two group. 

### Dimensionality Reduction

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

### Principal Component Analysis (PCA)

**PCA** is the most commonly used technique for dimensionality reduction. The first component produced in PCA comprises the majority of information or variance within the data. PCA uses a **covariance matrix** to measure the relationship between features of the dataset. The eigenvectors tell the directions of the spread of the data. The eigenvalues indicate the relative importance of these directions.

Let's see what it looks like with when using PCA in 1 dimension. 


```python
# Transform the data with only the first principal component
pca = PCA(n_components=1)

# Store the transformed data in the data_transformed
data_transformed = pca.fit_transform(data.values) 
```


```python
plt.figure(figsize=(10, 7))
plt.plot(data_transformed)
plt.xlabel('Sample')
plt.ylabel('Transformed Data')
plt.title('The dataset transformed by the principal component')
plt.show()
```


    
![png](output_120_0.png)
    


The transformed data are between -3 to 30, the transformed data value are going up and down in the one dimension.


```python
print("Original shape:   ", data.shape)
print("Transformed shape:", data_transformed.shape)
```

    Original shape:    (8949, 17)
    Transformed shape: (8949, 1)
    

**PCA in 2 Dimensions**


```python
# Transform the data with only the first principal component
pca2 = PCA(n_components=2)

# Store the transformed data in the data_transformed
data_pca2 = pca2.fit_transform(data.values) 
```


```python
print("Original shape:   ", data.shape)
print("Transformed shape:", data_pca2.shape)
```

    Original shape:    (8949, 17)
    Transformed shape: (8949, 2)
    


```python
data_pca2 = pd.DataFrame(data_pca2)
```


```python
data_pca2.iloc[:,0]
```




    0      -1.682361
    1      -1.138968
    2       0.969376
    3      -0.873814
    4      -1.599681
              ...   
    8944   -0.359267
    8945   -0.564022
    8946   -0.925785
    8947   -2.336210
    8948   -0.556041
    Name: 0, Length: 8949, dtype: float64



Let's check what it looks like with a k-means clustering of n_clusters =8. 


```python
plt.scatter(data_pca2.iloc[:,0],data_pca2.iloc[:,1],
            c = KMeans(n_clusters=8).fit_predict(data_pca2), cmap =None) 
plt.show()
```


    
![png](output_129_0.png)
    


Looks like it has better performance to classify customers into 8 groups with PCA method. 

#### PCA K-means

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python
km_list_pca = []
for i in range (2,11):
    km = KMeans(i)
    km_clusters = km.fit_predict(data_pca2)
    sil_score = silhouette_score(data_pca2, km_clusters)
    print(f"k={i} K-Means Clustering: {sil_score}")
    
    km_list_pca.append((i, sil_score))
    
    plt.scatter(data_pca2.iloc[:,0],data_pca2.iloc[:,1], c = km_clusters, cmap =None)
    plt.title(f"Customer Segmentation with K-means clusters when k={i}")
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

df_km_pca = pd.DataFrame(km_list_pca, columns=['k', 'silhouette_score'])
```

    k=2 K-Means Clustering: 0.4648388895825171
    


    
![png](output_132_1.png)
    


    k=3 K-Means Clustering: 0.4522981030357994
    


    
![png](output_132_3.png)
    


    k=4 K-Means Clustering: 0.40763316701596436
    


    
![png](output_132_5.png)
    


    k=5 K-Means Clustering: 0.4010285069965804
    


    
![png](output_132_7.png)
    


    k=6 K-Means Clustering: 0.3832113480741601
    


    
![png](output_132_9.png)
    


    k=7 K-Means Clustering: 0.37864676700187855
    


    
![png](output_132_11.png)
    


    k=8 K-Means Clustering: 0.3938322095862114
    


    
![png](output_132_13.png)
    


    k=9 K-Means Clustering: 0.371752400854586
    


    
![png](output_132_15.png)
    


    k=10 K-Means Clustering: 0.3644457062541696
    


    
![png](output_132_17.png)
    



```python
df_km_pca.sort_values('silhouette_score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.464839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.452298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.407633</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.401029</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.393832</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.383211</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.378647</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>0.371752</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>0.364446</td>
    </tr>
  </tbody>
</table>
</div>



Compare with the k-means without PCA scaled, the silhouette_score of PCA using the k-means method is much **better**. The best one is when k is equal to **2**. 

#### PCA Hierarchical Clustering

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python
ac_list_pca = []

for i in range (2,11):
    for linkage_method in ['single', 'ward', 'average', 'complete']:
        ac = AgglomerativeClustering(n_clusters=i, linkage=linkage_method)
        ac_clusters = ac.fit_predict(data_pca2)
        sil_score = silhouette_score(data_pca2, ac_clusters)
        print(f"n_clusters={i}, linkage={linkage_method}   Agglomerative Clustering: {sil_score}")
        
        ac_list_pca.append((i, linkage_method, sil_score))

        plt.scatter(data_pca2.iloc[:,0],data_pca2.iloc[:,1], c = ac_clusters, cmap =None)
        plt.title(f"Customer Segmentation with Agglomerative clusters (n_clusters={i}, linkage={linkage_method})")
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()

df_ac_pca = pd.DataFrame(ac_list_pca, columns=['number_of_clusters', 'linkage_method', 'silhouette_score'])
```

    n_clusters=2, linkage=single   Agglomerative Clustering: 0.8736727809002071
    


    
![png](output_136_1.png)
    


    n_clusters=2, linkage=ward   Agglomerative Clustering: 0.3587633210425497
    


    
![png](output_136_3.png)
    


    n_clusters=2, linkage=average   Agglomerative Clustering: 0.8736727809002071
    


    
![png](output_136_5.png)
    


    n_clusters=2, linkage=complete   Agglomerative Clustering: 0.8801175146313748
    


    
![png](output_136_7.png)
    


    n_clusters=3, linkage=single   Agglomerative Clustering: 0.8717912306703616
    


    
![png](output_136_9.png)
    


    n_clusters=3, linkage=ward   Agglomerative Clustering: 0.35824510887129646
    


    
![png](output_136_11.png)
    


    n_clusters=3, linkage=average   Agglomerative Clustering: 0.8277887570388014
    


    
![png](output_136_13.png)
    


    n_clusters=3, linkage=complete   Agglomerative Clustering: 0.8699532963877434
    


    
![png](output_136_15.png)
    


    n_clusters=4, linkage=single   Agglomerative Clustering: 0.8714914550534195
    


    
![png](output_136_17.png)
    


    n_clusters=4, linkage=ward   Agglomerative Clustering: 0.38129954855712545
    


    
![png](output_136_19.png)
    


    n_clusters=4, linkage=average   Agglomerative Clustering: 0.8068584504607084
    


    
![png](output_136_21.png)
    


    n_clusters=4, linkage=complete   Agglomerative Clustering: 0.6811901887559892
    


    
![png](output_136_23.png)
    


    n_clusters=5, linkage=single   Agglomerative Clustering: 0.8660871239972211
    


    
![png](output_136_25.png)
    


    n_clusters=5, linkage=ward   Agglomerative Clustering: 0.3606714677079056
    


    
![png](output_136_27.png)
    


    n_clusters=5, linkage=average   Agglomerative Clustering: 0.8023278062582475
    


    
![png](output_136_29.png)
    


    n_clusters=5, linkage=complete   Agglomerative Clustering: 0.6247620202642238
    


    
![png](output_136_31.png)
    


    n_clusters=6, linkage=single   Agglomerative Clustering: 0.8632278551073872
    


    
![png](output_136_33.png)
    


    n_clusters=6, linkage=ward   Agglomerative Clustering: 0.36258552135429817
    


    
![png](output_136_35.png)
    


    n_clusters=6, linkage=average   Agglomerative Clustering: 0.646101502384973
    


    
![png](output_136_37.png)
    


    n_clusters=6, linkage=complete   Agglomerative Clustering: 0.6119824770392213
    


    
![png](output_136_39.png)
    


    n_clusters=7, linkage=single   Agglomerative Clustering: 0.8176192529468731
    


    
![png](output_136_41.png)
    


    n_clusters=7, linkage=ward   Agglomerative Clustering: 0.3100908065829832
    


    
![png](output_136_43.png)
    


    n_clusters=7, linkage=average   Agglomerative Clustering: 0.6170684089927188
    


    
![png](output_136_45.png)
    


    n_clusters=7, linkage=complete   Agglomerative Clustering: 0.5921934386694968
    


    
![png](output_136_47.png)
    


    n_clusters=8, linkage=single   Agglomerative Clustering: 0.8167737181751579
    


    
![png](output_136_49.png)
    


    n_clusters=8, linkage=ward   Agglomerative Clustering: 0.31501033354873675
    


    
![png](output_136_51.png)
    


    n_clusters=8, linkage=average   Agglomerative Clustering: 0.6168535959853301
    


    
![png](output_136_53.png)
    


    n_clusters=8, linkage=complete   Agglomerative Clustering: 0.5805347222215641
    


    
![png](output_136_55.png)
    


    n_clusters=9, linkage=single   Agglomerative Clustering: 0.8067701179465733
    


    
![png](output_136_57.png)
    


    n_clusters=9, linkage=ward   Agglomerative Clustering: 0.3129757522613688
    


    
![png](output_136_59.png)
    


    n_clusters=9, linkage=average   Agglomerative Clustering: 0.6021485822531096
    


    
![png](output_136_61.png)
    


    n_clusters=9, linkage=complete   Agglomerative Clustering: 0.37487250128948296
    


    
![png](output_136_63.png)
    


    n_clusters=10, linkage=single   Agglomerative Clustering: 0.7643125817139251
    


    
![png](output_136_65.png)
    


    n_clusters=10, linkage=ward   Agglomerative Clustering: 0.3143232591571389
    


    
![png](output_136_67.png)
    


    n_clusters=10, linkage=average   Agglomerative Clustering: 0.582341633615677
    


    
![png](output_136_69.png)
    


    n_clusters=10, linkage=complete   Agglomerative Clustering: 0.37253140969822063
    


    
![png](output_136_71.png)
    



```python
df_ac_pca.sort_values('silhouette_score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_clusters</th>
      <th>linkage_method</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>complete</td>
      <td>0.880118</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>single</td>
      <td>0.873673</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>average</td>
      <td>0.873673</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>single</td>
      <td>0.871791</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>single</td>
      <td>0.871491</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>complete</td>
      <td>0.869953</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>single</td>
      <td>0.866087</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6</td>
      <td>single</td>
      <td>0.863228</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>average</td>
      <td>0.827789</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>single</td>
      <td>0.817619</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8</td>
      <td>single</td>
      <td>0.816774</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>average</td>
      <td>0.806858</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9</td>
      <td>single</td>
      <td>0.806770</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>average</td>
      <td>0.802328</td>
    </tr>
    <tr>
      <th>32</th>
      <td>10</td>
      <td>single</td>
      <td>0.764313</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>complete</td>
      <td>0.681190</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6</td>
      <td>average</td>
      <td>0.646102</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>complete</td>
      <td>0.624762</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>average</td>
      <td>0.617068</td>
    </tr>
    <tr>
      <th>26</th>
      <td>8</td>
      <td>average</td>
      <td>0.616854</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>complete</td>
      <td>0.611982</td>
    </tr>
    <tr>
      <th>30</th>
      <td>9</td>
      <td>average</td>
      <td>0.602149</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>complete</td>
      <td>0.592193</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10</td>
      <td>average</td>
      <td>0.582342</td>
    </tr>
    <tr>
      <th>27</th>
      <td>8</td>
      <td>complete</td>
      <td>0.580535</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>ward</td>
      <td>0.381300</td>
    </tr>
    <tr>
      <th>31</th>
      <td>9</td>
      <td>complete</td>
      <td>0.374873</td>
    </tr>
    <tr>
      <th>35</th>
      <td>10</td>
      <td>complete</td>
      <td>0.372531</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>ward</td>
      <td>0.362586</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>ward</td>
      <td>0.360671</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ward</td>
      <td>0.358763</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>ward</td>
      <td>0.358245</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8</td>
      <td>ward</td>
      <td>0.315010</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10</td>
      <td>ward</td>
      <td>0.314323</td>
    </tr>
    <tr>
      <th>29</th>
      <td>9</td>
      <td>ward</td>
      <td>0.312976</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>ward</td>
      <td>0.310091</td>
    </tr>
  </tbody>
</table>
</div>



After comparing with the plots and the table above, the **ward** linkage method seems to have a better distribution of clusters. The ward methods with **4 clusters** have the highest silhouette score. 


```python
plt.figure(figsize=(12, 5))
dendrogram = sch.dendrogram(sch.linkage(data_pca2, method = 'ward'))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](output_139_0.png)
    


We can see that the Dendrogram of ward method from PCA generated a much clear relationship. 

#### PCA DBSCAN

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python
db_list_pca = []

#Evaluate DBSCAN hyperparameters and their effect on the silhouette score
for ep in np.arange(1, 3, 0.5):
    for min_sample in range(2, 20, 4):
        db = DBSCAN(eps=ep, min_samples = min_sample)
        db_clusters = db.fit_predict(data_pca2)
        sil_score = silhouette_score(data_pca2, db_clusters)
        db_list_pca.append((ep, min_sample, sil_score, len(set(db.labels_))))

        plt.scatter(data_pca2.iloc[:,0],data_pca2.iloc[:,1], c = db_clusters, cmap =None)
        plt.title('Customer Segmentation with DBSCAN Epsilon: ' + str(ep) + ' | Minimum Points: ' + str(min_sample))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()

        print("Silhouette Score: ", sil_score)
```


    
![png](output_142_0.png)
    


    Silhouette Score:  0.642499625696301
    


    
![png](output_142_2.png)
    


    Silhouette Score:  0.5925208754872522
    


    
![png](output_142_4.png)
    


    Silhouette Score:  0.765579627806045
    


    
![png](output_142_6.png)
    


    Silhouette Score:  0.750275015456107
    


    
![png](output_142_8.png)
    


    Silhouette Score:  0.7426116301831411
    


    
![png](output_142_10.png)
    


    Silhouette Score:  0.7206871883099984
    


    
![png](output_142_12.png)
    


    Silhouette Score:  0.781142604248167
    


    
![png](output_142_14.png)
    


    Silhouette Score:  0.7945355883556591
    


    
![png](output_142_16.png)
    


    Silhouette Score:  0.7902202747746846
    


    
![png](output_142_18.png)
    


    Silhouette Score:  0.788046736289255
    


    
![png](output_142_20.png)
    


    Silhouette Score:  0.7622827920138017
    


    
![png](output_142_22.png)
    


    Silhouette Score:  0.8032524907791072
    


    
![png](output_142_24.png)
    


    Silhouette Score:  0.809185587854294
    


    
![png](output_142_26.png)
    


    Silhouette Score:  0.8060666941194006
    


    
![png](output_142_28.png)
    


    Silhouette Score:  0.803340681555949
    


    
![png](output_142_30.png)
    


    Silhouette Score:  0.8628676027107722
    


    
![png](output_142_32.png)
    


    Silhouette Score:  0.8474518419210371
    


    
![png](output_142_34.png)
    


    Silhouette Score:  0.8079078681176542
    


    
![png](output_142_36.png)
    


    Silhouette Score:  0.8254894849343388
    


    
![png](output_142_38.png)
    


    Silhouette Score:  0.8192426451642665
    


```python
df_db_pca = pd.DataFrame(db_list_pca, columns=['Epsilon', 'Minimum Sample', 'Silhouette Score', 'Number of clusters'])
df_db_pca
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epsilon</th>
      <th>Minimum Sample</th>
      <th>Silhouette Score</th>
      <th>Number of clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2</td>
      <td>0.642500</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>6</td>
      <td>0.592521</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>10</td>
      <td>0.765580</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>14</td>
      <td>0.750275</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>18</td>
      <td>0.742612</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>2</td>
      <td>0.720687</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.5</td>
      <td>6</td>
      <td>0.781143</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.5</td>
      <td>10</td>
      <td>0.794536</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.5</td>
      <td>14</td>
      <td>0.790220</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.5</td>
      <td>18</td>
      <td>0.788047</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.0</td>
      <td>2</td>
      <td>0.762283</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2.0</td>
      <td>6</td>
      <td>0.803252</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.0</td>
      <td>10</td>
      <td>0.809186</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.0</td>
      <td>14</td>
      <td>0.806067</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.0</td>
      <td>18</td>
      <td>0.803341</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.5</td>
      <td>2</td>
      <td>0.862868</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.5</td>
      <td>6</td>
      <td>0.847452</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.5</td>
      <td>10</td>
      <td>0.807908</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.5</td>
      <td>14</td>
      <td>0.825489</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.5</td>
      <td>18</td>
      <td>0.819243</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_db_pca.sort_values(by=['Silhouette Score'], ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epsilon</th>
      <th>Minimum Sample</th>
      <th>Silhouette Score</th>
      <th>Number of clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>2.5</td>
      <td>2</td>
      <td>0.862868</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.5</td>
      <td>6</td>
      <td>0.847452</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.5</td>
      <td>14</td>
      <td>0.825489</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.5</td>
      <td>18</td>
      <td>0.819243</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.0</td>
      <td>10</td>
      <td>0.809186</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



After comparing with the plots and the table above, the **eps = 2.5 and min_samples = 18** seem to generate a better performance with two clusters.

#### T-Distributed Stochastic Neighbor Embedding (TSNE)

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

**TSNE** is also an unsupervised non-linear dimensionality reduction technique. The **t-distribution** is used when dealing with a small sample size with an unknown population standard deviation. 


```python
model_tsne = TSNE(n_components=2, verbose=1)
```


```python
data_tsne = model_tsne.fit_transform(data)
```

    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.042s...
    [t-SNE] Computed neighbors for 8949 samples in 1.916s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.427252
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 83.966179
    [t-SNE] KL divergence after 1000 iterations: 1.505604
    

Let's check what it looks like with a k-means clustering of n_clusters =8.


```python
data_tsne = pd.DataFrame(data_tsne)

plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1],
            c = KMeans(n_clusters=8).fit_predict(data_tsne), cmap =None) 
plt.show()
```


    
![png](output_151_0.png)
    


Looks like TSNE has a better performance on scaling the data into two dimensions. 


```python
perplexity_values = [1, 5, 20, 30, 40, 60, 80, 400]
for perp in perplexity_values:
    model_tsne = TSNE(verbose=1, perplexity=perp)
    data_tsne = model_tsne.fit_transform(data)
    
    data_tsne = pd.DataFrame(data_tsne)
    
    plt.title(f'Low Dimensional Representation of Customer Segmentation. Perplexity {perp}');
    plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1], c = KMeans(3).fit_predict(data_pca2), cmap =None) 
    plt.figure(figsize=(10, 7))
    
    
plt.show()
```

    [t-SNE] Computing 4 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.040s...
    [t-SNE] Computed neighbors for 8949 samples in 0.841s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.031649
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 98.881966
    [t-SNE] KL divergence after 1000 iterations: 1.832245
    [t-SNE] Computing 16 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.037s...
    [t-SNE] Computed neighbors for 8949 samples in 1.140s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.208148
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 94.784729
    [t-SNE] KL divergence after 1000 iterations: 1.638389
    [t-SNE] Computing 61 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.030s...
    [t-SNE] Computed neighbors for 8949 samples in 1.496s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.371830
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 87.006981
    [t-SNE] KL divergence after 1000 iterations: 1.553980
    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.029s...
    [t-SNE] Computed neighbors for 8949 samples in 1.696s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.427252
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 83.965981
    [t-SNE] KL divergence after 1000 iterations: 1.499685
    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.030s...
    [t-SNE] Computed neighbors for 8949 samples in 1.820s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.469249
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 81.708488
    [t-SNE] KL divergence after 1000 iterations: 1.451017
    [t-SNE] Computing 181 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.032s...
    [t-SNE] Computed neighbors for 8949 samples in 2.230s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.534056
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 78.434349
    [t-SNE] KL divergence after 1000 iterations: 1.374042
    [t-SNE] Computing 241 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.056s...
    [t-SNE] Computed neighbors for 8949 samples in 3.360s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.583147
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 76.077538
    [t-SNE] KL divergence after 1000 iterations: 1.310628
    [t-SNE] Computing 1201 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.035s...
    [t-SNE] Computed neighbors for 8949 samples in 5.361s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.947532
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 62.363239
    [t-SNE] KL divergence after 1000 iterations: 0.869970
    


    
![png](output_153_1.png)
    



    
![png](output_153_2.png)
    



    
![png](output_153_3.png)
    



    
![png](output_153_4.png)
    



    
![png](output_153_5.png)
    



    
![png](output_153_6.png)
    



    
![png](output_153_7.png)
    



    
![png](output_153_8.png)
    



    <Figure size 1000x700 with 0 Axes>


From the above plots, most of the data points are at the center of the plot when the perplexity value is equal to 1. The plot is hard to identify any patterns and clusters when the perplexity is equal to 1. When the perplexity value increase, the relationship of clusters is getting clear. However, the perplexity value of 400 is seems to be too high. 

Let's check how it perform with a k-means clustering of n_clusters =8:


```python
perplexity_values = [1, 5, 20, 30, 40, 60, 80, 400]
for perp in perplexity_values:
    model_tsne = TSNE(verbose=1, perplexity=perp)
    data_tsne = model_tsne.fit_transform(data)
    
    data_tsne = pd.DataFrame(data_tsne)
    
    plt.title(f'Low Dimensional Representation of Customer Segmentation. Perplexity {perp}');
    plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1], c = KMeans(n_clusters=8).fit_predict(data_tsne), cmap =None) 
    plt.figure(figsize=(10, 7))
    
    
plt.show()
```

    [t-SNE] Computing 4 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.031s...
    [t-SNE] Computed neighbors for 8949 samples in 0.771s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.031649
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 98.866287
    [t-SNE] KL divergence after 1000 iterations: 1.833596
    [t-SNE] Computing 16 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.029s...
    [t-SNE] Computed neighbors for 8949 samples in 1.154s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.208148
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 94.692970
    [t-SNE] KL divergence after 1000 iterations: 1.629820
    [t-SNE] Computing 61 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.029s...
    [t-SNE] Computed neighbors for 8949 samples in 1.833s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.371830
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 87.428566
    [t-SNE] KL divergence after 1000 iterations: 1.548754
    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.032s...
    [t-SNE] Computed neighbors for 8949 samples in 2.160s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.427252
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 83.966995
    [t-SNE] KL divergence after 1000 iterations: 1.499196
    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.028s...
    [t-SNE] Computed neighbors for 8949 samples in 1.888s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.469249
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 82.194923
    [t-SNE] KL divergence after 1000 iterations: 1.449479
    [t-SNE] Computing 181 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.036s...
    [t-SNE] Computed neighbors for 8949 samples in 2.517s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.534056
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 78.441399
    [t-SNE] KL divergence after 1000 iterations: 1.372881
    [t-SNE] Computing 241 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.029s...
    [t-SNE] Computed neighbors for 8949 samples in 2.395s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.583147
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 76.082741
    [t-SNE] KL divergence after 1000 iterations: 1.310130
    [t-SNE] Computing 1201 nearest neighbors...
    [t-SNE] Indexed 8949 samples in 0.030s...
    [t-SNE] Computed neighbors for 8949 samples in 4.380s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 8949
    [t-SNE] Computed conditional probabilities for sample 2000 / 8949
    [t-SNE] Computed conditional probabilities for sample 3000 / 8949
    [t-SNE] Computed conditional probabilities for sample 4000 / 8949
    [t-SNE] Computed conditional probabilities for sample 5000 / 8949
    [t-SNE] Computed conditional probabilities for sample 6000 / 8949
    [t-SNE] Computed conditional probabilities for sample 7000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8000 / 8949
    [t-SNE] Computed conditional probabilities for sample 8949 / 8949
    [t-SNE] Mean sigma: 0.947532
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 62.362911
    [t-SNE] KL divergence after 1000 iterations: 0.869700
    


    
![png](output_156_1.png)
    



    
![png](output_156_2.png)
    



    
![png](output_156_3.png)
    



    
![png](output_156_4.png)
    



    
![png](output_156_5.png)
    



    
![png](output_156_6.png)
    



    
![png](output_156_7.png)
    



    
![png](output_156_8.png)
    



    <Figure size 1000x700 with 0 Axes>


From the above plots, most of the data points are at the center of the plot when the perplexity value is equal to 1. The data points are still too close to the middle of the plot when the perplexity value is equal to 5. The plots of 20 to 60 seem to generate a clear relationship. Therefore, we choose the default perplexity value of 30 for the TSNE model. 

#### TSNE K-means

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python
km_list_tsne = []
for i in range (2,11):
    km = KMeans(i)
    km_clusters = km.fit_predict(data_tsne)
    sil_score = silhouette_score(data_tsne, km_clusters)
    print(f"k={i} K-Means Clustering: {sil_score}")
    
    km_list_tsne.append((i, sil_score))
    
    plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1], c = km_clusters, cmap =None)
    plt.title(f"Customer Segmentation with K-means clusters when k={i}")
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

df_km_tsne = pd.DataFrame(km_list_tsne, columns=['k', 'silhouette_score'])
```

    k=2 K-Means Clustering: 0.4067707061767578
    


    
![png](output_159_1.png)
    


    k=3 K-Means Clustering: 0.4468262791633606
    


    
![png](output_159_3.png)
    


    k=4 K-Means Clustering: 0.43460649251937866
    


    
![png](output_159_5.png)
    


    k=5 K-Means Clustering: 0.4032849967479706
    


    
![png](output_159_7.png)
    


    k=6 K-Means Clustering: 0.4076898694038391
    


    
![png](output_159_9.png)
    


    k=7 K-Means Clustering: 0.41257354617118835
    


    
![png](output_159_11.png)
    


    k=8 K-Means Clustering: 0.4125075042247772
    


    
![png](output_159_13.png)
    


    k=9 K-Means Clustering: 0.41203945875167847
    


    
![png](output_159_15.png)
    


    k=10 K-Means Clustering: 0.4134407043457031
    


    
![png](output_159_17.png)
    



```python
df_km_tsne.sort_values('silhouette_score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.446826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.434606</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>0.413441</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.412574</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.412508</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>0.412039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.407690</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.406771</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.403285</td>
    </tr>
  </tbody>
</table>
</div>



The k-means clustering with tsne scaled data seems to have the best performance when **k is equal to 3**. However, compared with pca scaled data with k-means, the silhouette_score is a little bit lower. 

#### TSNE Hierarchical Clustering

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python
ac_list_tsne = []

for i in range (2,11):
    for linkage_method in ['single', 'ward', 'average', 'complete']:
        ac = AgglomerativeClustering(n_clusters=i, linkage=linkage_method)
        ac_clusters = ac.fit_predict(data_tsne)
        sil_score = silhouette_score(data_tsne, ac_clusters)
        print(f"n_clusters={i}, linkage={linkage_method}   Agglomerative Clustering: {sil_score}")
        
        ac_list_tsne.append((i, linkage_method, sil_score))

        plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1], c = ac_clusters, cmap =None)
        plt.title(f"Customer Segmentation with Agglomerative clusters (n_clusters={i}, linkage={linkage_method})")
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()

df_ac_tsne = pd.DataFrame(ac_list_tsne, columns=['number_of_clusters', 'linkage_method', 'silhouette_score'])
```

    n_clusters=2, linkage=single   Agglomerative Clustering: 0.06704951077699661
    


    
![png](output_163_1.png)
    


    n_clusters=2, linkage=ward   Agglomerative Clustering: 0.3696958124637604
    


    
![png](output_163_3.png)
    


    n_clusters=2, linkage=average   Agglomerative Clustering: 0.3784180283546448
    


    
![png](output_163_5.png)
    


    n_clusters=2, linkage=complete   Agglomerative Clustering: 0.40311089158058167
    


    
![png](output_163_7.png)
    


    n_clusters=3, linkage=single   Agglomerative Clustering: -0.1539221554994583
    


    
![png](output_163_9.png)
    


    n_clusters=3, linkage=ward   Agglomerative Clustering: 0.42572563886642456
    


    
![png](output_163_11.png)
    


    n_clusters=3, linkage=average   Agglomerative Clustering: 0.42108476161956787
    


    
![png](output_163_13.png)
    


    n_clusters=3, linkage=complete   Agglomerative Clustering: 0.4141015410423279
    


    
![png](output_163_15.png)
    


    n_clusters=4, linkage=single   Agglomerative Clustering: -0.3698209524154663
    


    
![png](output_163_17.png)
    


    n_clusters=4, linkage=ward   Agglomerative Clustering: 0.38199636340141296
    


    
![png](output_163_19.png)
    


    n_clusters=4, linkage=average   Agglomerative Clustering: 0.3599488139152527
    


    
![png](output_163_21.png)
    


    n_clusters=4, linkage=complete   Agglomerative Clustering: 0.37650686502456665
    


    
![png](output_163_23.png)
    


    n_clusters=5, linkage=single   Agglomerative Clustering: -0.3893791139125824
    


    
![png](output_163_25.png)
    


    n_clusters=5, linkage=ward   Agglomerative Clustering: 0.3474059998989105
    


    
![png](output_163_27.png)
    


    n_clusters=5, linkage=average   Agglomerative Clustering: 0.3440847098827362
    


    
![png](output_163_29.png)
    


    n_clusters=5, linkage=complete   Agglomerative Clustering: 0.37016379833221436
    


    
![png](output_163_31.png)
    


    n_clusters=6, linkage=single   Agglomerative Clustering: -0.45555752515792847
    


    
![png](output_163_33.png)
    


    n_clusters=6, linkage=ward   Agglomerative Clustering: 0.362177312374115
    


    
![png](output_163_35.png)
    


    n_clusters=6, linkage=average   Agglomerative Clustering: 0.35542333126068115
    


    
![png](output_163_37.png)
    


    n_clusters=6, linkage=complete   Agglomerative Clustering: 0.34130820631980896
    


    
![png](output_163_39.png)
    


    n_clusters=7, linkage=single   Agglomerative Clustering: -0.4936814308166504
    


    
![png](output_163_41.png)
    


    n_clusters=7, linkage=ward   Agglomerative Clustering: 0.3497436046600342
    


    
![png](output_163_43.png)
    


    n_clusters=7, linkage=average   Agglomerative Clustering: 0.3562188744544983
    


    
![png](output_163_45.png)
    


    n_clusters=7, linkage=complete   Agglomerative Clustering: 0.3771764934062958
    


    
![png](output_163_47.png)
    


    n_clusters=8, linkage=single   Agglomerative Clustering: -0.5054686069488525
    


    
![png](output_163_49.png)
    


    n_clusters=8, linkage=ward   Agglomerative Clustering: 0.35989323258399963
    


    
![png](output_163_51.png)
    


    n_clusters=8, linkage=average   Agglomerative Clustering: 0.34580090641975403
    


    
![png](output_163_53.png)
    


    n_clusters=8, linkage=complete   Agglomerative Clustering: 0.36205610632896423
    


    
![png](output_163_55.png)
    


    n_clusters=9, linkage=single   Agglomerative Clustering: -0.511385440826416
    


    
![png](output_163_57.png)
    


    n_clusters=9, linkage=ward   Agglomerative Clustering: 0.36603590846061707
    


    
![png](output_163_59.png)
    


    n_clusters=9, linkage=average   Agglomerative Clustering: 0.3526369333267212
    


    
![png](output_163_61.png)
    


    n_clusters=9, linkage=complete   Agglomerative Clustering: 0.36377862095832825
    


    
![png](output_163_63.png)
    


    n_clusters=10, linkage=single   Agglomerative Clustering: -0.5562132000923157
    


    
![png](output_163_65.png)
    


    n_clusters=10, linkage=ward   Agglomerative Clustering: 0.357479453086853
    


    
![png](output_163_67.png)
    


    n_clusters=10, linkage=average   Agglomerative Clustering: 0.35839247703552246
    


    
![png](output_163_69.png)
    


    n_clusters=10, linkage=complete   Agglomerative Clustering: 0.3551463484764099
    


    
![png](output_163_71.png)
    



```python
df_ac_tsne.sort_values('silhouette_score', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number_of_clusters</th>
      <th>linkage_method</th>
      <th>silhouette_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>ward</td>
      <td>0.425726</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>average</td>
      <td>0.421085</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>complete</td>
      <td>0.414102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>complete</td>
      <td>0.403111</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>ward</td>
      <td>0.381996</td>
    </tr>
  </tbody>
</table>
</div>



After comparing with the top five silhouette_score and the above plots, the linkage_method of **ward** with number_of_clusters of **3** has the best performance. The score is a better than the PCA method.


```python
plt.figure(figsize=(12, 5))
dendrogram = sch.dendrogram(sch.linkage(data_tsne, method = 'ward'))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](output_166_0.png)
    


We can see that the Dendrogram of ward method from TSNE generated a clear relationship.

#### TSNE DBSCAN

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python
db_list_tsne = []

#Evaluate DBSCAN hyperparameters and their effect on the silhouette score
for ep in np.arange(1.0, 2.5, 0.5):
    for min_sample in range(10, 40, 4):
        db = DBSCAN(eps=ep, min_samples = min_sample)
        db_clusters = db.fit_predict(data_tsne)
        sil_score = silhouette_score(data_tsne, db_clusters)
        db_list_tsne.append((ep, min_sample, sil_score, len(set(db.labels_))))

        plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1], c = db_clusters, cmap =None)
        plt.title('Customer Segmentation with DBSCAN Epsilon: ' + str(ep) + ' | Minimum Points: ' + str(min_sample))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()

        print("Silhouette Score: ", sil_score)

df_db_tsne = pd.DataFrame(db_list_tsne, columns=['Epsilon', 'Minimum Sample', 'Silhouette Score', 'Number of clusters'])
```


    
![png](output_169_0.png)
    


    Silhouette Score:  -0.42703268
    


    
![png](output_169_2.png)
    


    Silhouette Score:  -0.32993916
    


    
![png](output_169_4.png)
    


    Silhouette Score:  -0.40936393
    


    
![png](output_169_6.png)
    


    Silhouette Score:  -0.123597905
    


    
![png](output_169_8.png)
    


    Silhouette Score:  -0.08758138
    


    
![png](output_169_10.png)
    


    Silhouette Score:  0.032062296
    


    
![png](output_169_12.png)
    


    Silhouette Score:  -0.032074798
    


    
![png](output_169_14.png)
    


    Silhouette Score:  -0.14196473
    


    
![png](output_169_16.png)
    


    Silhouette Score:  0.03737406
    


    
![png](output_169_18.png)
    


    Silhouette Score:  -0.010809418
    


    
![png](output_169_20.png)
    


    Silhouette Score:  -0.0060092383
    


    
![png](output_169_22.png)
    


    Silhouette Score:  -0.03272851
    


    
![png](output_169_24.png)
    


    Silhouette Score:  -0.11420548
    


    
![png](output_169_26.png)
    


    Silhouette Score:  0.11205335
    


    
![png](output_169_28.png)
    


    Silhouette Score:  -0.084032446
    


    
![png](output_169_30.png)
    


    Silhouette Score:  -0.20862171
    


    
![png](output_169_32.png)
    


    Silhouette Score:  0.10873683
    


    
![png](output_169_34.png)
    


    Silhouette Score:  0.06580043
    


    
![png](output_169_36.png)
    


    Silhouette Score:  0.06714135
    


    
![png](output_169_38.png)
    


    Silhouette Score:  0.018478423
    


    
![png](output_169_40.png)
    


    Silhouette Score:  0.008669822
    


    
![png](output_169_42.png)
    


    Silhouette Score:  -0.01404849
    


    
![png](output_169_44.png)
    


    Silhouette Score:  -0.040791847
    


    
![png](output_169_46.png)
    


    Silhouette Score:  -0.032392904
    


```python
df_db_tsne.sort_values(by=['Silhouette Score'], ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epsilon</th>
      <th>Minimum Sample</th>
      <th>Silhouette Score</th>
      <th>Number of clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>1.5</td>
      <td>30</td>
      <td>0.112053</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.0</td>
      <td>10</td>
      <td>0.108737</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.0</td>
      <td>18</td>
      <td>0.067141</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.0</td>
      <td>14</td>
      <td>0.065800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.5</td>
      <td>10</td>
      <td>0.037374</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



From the above table, we can see that the silhouette scores are low with tsne. 

### Discussion

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)

Let's create a summary table with the best silhouette_score from each method.

|Model | # of clusters | linkage_method | eps | min_samples | silhouette_score |
|-----: | :-: |:-: |:-: |:-: |:-: |
| kmean | 3	| - | - | - | 0.2505 |
| ac | 2 | complete| - | -| 0.7865 |
| dbscan | 2 | - | 2.5 | 10 | 0.5541 | 
| km_pca_2 | 2 | - | - | - | 0.4648 | 
| <strong>km_pca_3</strong> | <strong>3</strong> | <strong>-</strong> | <strong>-</strong> | <strong>-</strong> | <strong>0.4523</strong> | 
| ac_pca | 4 | ward | - | - | 0.3813 | 
| dbscan_pca | 2 | - | 2.5	 | 18 | 0.8192 | 
| km_tsne | 3 | - | - | - | 0.4468 | 
| ac_tsne | 3 | ward | - | - | 0.4257 | 
| dbscan_tsne | 4 | - | 1.5	 | 30 | 0.1121 | 

From the table above, the silhouette score is highest when the number of clusters is equal to two. However, after comparing with the plots that we generated before, 3 clusters can help us get better insights from the data. Moreover, most methods with 3 clusters are able to generate a desired silhouette_score. Since the model **km_pca_3** has the highest silhouette_score, it is chosen to be the best model from the above analysis. The silhouette score of 0.4523 is desirable. 

Let's visualize the detail performance of the model **km_pca_3**.


```python
km_pca_3
```




    array([0, 2, 0, ..., 0, 0, 0])



The clusters are labelled as 0, 1, and 2. 


```python
print(f"k=3 K-Means Clustering: {silhouette_score(data_pca2, km_pca_3)}")
```

    k=3 K-Means Clustering: 0.45231837920227425
    

Using PCA to visualization pca scaled data with k-means of n_clusters=3.


```python
plt.scatter(data_pca2.iloc[:,0],data_pca2.iloc[:,1], c = km_pca_3, cmap =None) 
plt.show()
```


    
![png](output_181_0.png)
    


Using TSNE to visualization pca scaled data with k-means of n_clusters=3:


```python
km_pca_3 = KMeans(3).fit_predict(data_pca2)
print(f"k=3 K-Means Clustering: {silhouette_score(data_pca2, km_pca_3)}")
#labels_km_pca_3 = km_pca_3.labels_
plt.scatter(data_tsne.iloc[:,0],data_tsne.iloc[:,1], c = km_pca_3, cmap =None) 
plt.show()
```

    k=3 K-Means Clustering: 0.45231837920227425
    


    
![png](output_183_1.png)
    


Create a new dataframe to combine clusters with the original data.


```python
df_km_pca_3 = pd.concat([df.reset_index(drop=True), pd.DataFrame({'cluster':km_pca_3}).reset_index(drop=True)], axis=1)
```


```python
df_km_pca_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.40</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.00</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>864.304943</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8944</th>
      <td>28.493517</td>
      <td>1.000000</td>
      <td>291.12</td>
      <td>0.00</td>
      <td>291.12</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>325.594462</td>
      <td>48.886365</td>
      <td>0.500000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8945</th>
      <td>19.183215</td>
      <td>1.000000</td>
      <td>300.00</td>
      <td>0.00</td>
      <td>300.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>6</td>
      <td>1000.0</td>
      <td>275.861322</td>
      <td>864.304943</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8946</th>
      <td>23.398673</td>
      <td>0.833333</td>
      <td>144.40</td>
      <td>0.00</td>
      <td>144.40</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.000000</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0</td>
      <td>5</td>
      <td>1000.0</td>
      <td>81.270775</td>
      <td>82.418369</td>
      <td>0.250000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8947</th>
      <td>13.457564</td>
      <td>0.833333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>36.558778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>2</td>
      <td>0</td>
      <td>500.0</td>
      <td>52.549959</td>
      <td>55.755628</td>
      <td>0.250000</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8948</th>
      <td>372.708075</td>
      <td>0.666667</td>
      <td>1093.25</td>
      <td>1093.25</td>
      <td>0.00</td>
      <td>127.040008</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>2</td>
      <td>23</td>
      <td>1200.0</td>
      <td>63.165404</td>
      <td>88.288956</td>
      <td>0.000000</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>8949 rows × 18 columns</p>
</div>



Use countplot to count number of data within that cluster.


```python
sns.countplot(x='cluster', data=df_km_pca_3)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22aebcd9388>




    
![png](output_188_1.png)
    


There are larger amounts of custome are in cluster 0. 

Let's create some plots to see the distribution of different features for each cluster.


```python
for c in df_km_pca_3.drop(['cluster'],axis=1):
    grid= sns.FacetGrid(df_km_pca_3, col='cluster')
    grid= grid.map(plt.hist, c)
plt.show()
```


    
![png](output_191_0.png)
    



    
![png](output_191_1.png)
    



    
![png](output_191_2.png)
    



    
![png](output_191_3.png)
    



    
![png](output_191_4.png)
    



    
![png](output_191_5.png)
    



    
![png](output_191_6.png)
    



    
![png](output_191_7.png)
    



    
![png](output_191_8.png)
    



    
![png](output_191_9.png)
    



    
![png](output_191_10.png)
    



    
![png](output_191_11.png)
    



    
![png](output_191_12.png)
    



    
![png](output_191_13.png)
    



    
![png](output_191_14.png)
    



    
![png](output_191_15.png)
    



    
![png](output_191_16.png)
    



```python
sns.pairplot(df_km_pca_3, hue="cluster")
```




    <seaborn.axisgrid.PairGrid at 0x22aebda36c8>




    
![png](output_192_1.png)
    


Create plots focus on the important features.


```python
df_km_pca_3_tmp = df_km_pca_3[['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'cluster']]
df_km_pca_3_tmp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.900749</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.4</td>
      <td>0.000000</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3202.467416</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6442.945483</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2495.148862</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1666.670542</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
      <td>205.788017</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>817.714335</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_km_pca_3_tmp2 = pd.melt(df_km_pca_3_tmp, id_vars='cluster', var_name="value_name", value_name="value")
df_km_pca_3_tmp2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>value_name</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>BALANCE</td>
      <td>40.900749</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>BALANCE</td>
      <td>3202.467416</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>BALANCE</td>
      <td>2495.148862</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>BALANCE</td>
      <td>1666.670542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>BALANCE</td>
      <td>817.714335</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.catplot(data=df_km_pca_3_tmp2, x="cluster", y="value", hue="value_name", kind='bar', height=5, aspect=2)
```




    <seaborn.axisgrid.FacetGrid at 0x22ae4e57608>




    
![png](output_196_1.png)
    


Let's create a summary table for these three customer group:

|Cluster | Balance | Purchases | Oneoff_purchases | Installments Purchases | Cash Advance | Credit_limit | Payments | Insurance Product Recommendation |
|:-----: | :-: |:-: |:-: |:-: |:-: |:-: |:-: | :-: |
| Cluster 0 | low	| low | low | low | low | low | low | <strong> Saving Plan</strong> |
| Cluster 1 | medium | high| high | high | low | high | high | <strong> Wealth Management </strong> |
| Cluster 2 | high | low | low | low | high | high | high | <strong> Loan </strong> |

**Recommendation:**<br>

**Cluster 0:** Customers who have low balances, low credit limits, and low purchases. These customers could be low-income and don't likely spend too much on purchasing goods. We should offer a **saving plan** for them. <br>
**Cluster 1:** Customers who have high credit limits, high purchases, low cash advance, and high payments. These customers could be medium and high-income customers who are able to pay for their credit cards on time. They don't use cash advance too often; therefore, we should offer a **wealth management plan** for this group of customers. <br>
**Cluster 2:** Customers who have a high balance, low purchase, high cash advance, high credit limit, and high payments. Customers who use cash advance a lot is more likely to need a loan. Therefore, we should  offer a **loan plan** for this group of customers

### Conclusion

The study explored a range of different clustering algorithms such as k-means, hierarchical clustering, and DBSCAN. Standardization is useful for unsupervised models that require distance metrics. Different hyperparameters are evaluated with the silhouette score. The silhouette score is a metric that helps evaluate the performance of unsupervised learning methods. PCA and TSNE are methods used for dimensionality reduction and visualization in the project. After comparing with the silhouette score and visualized plots, '3' is the optimal number of clusters for the dataset. The PCA scaled data that used the k-means method with a k value of three is the optimal choice. 

Based on the above analysis, customers can be divided into three groups. The first group of customers are low-incomers and small spenders; therefore, a saving plan is recommended for this group. The second group of customers are able to pay for credit cards on time and don't like to use cash advance so the company should offer a wealth management plan for this group. The last group of customers who use cash advance a lot are more likely to accept a loan plan from the insurance company. 

[Return to top](#Final-Project:-Applied-Unsupervised-Learning)


```python

```
