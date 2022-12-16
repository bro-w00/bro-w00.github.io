---
layout: post
title: Pycaret Juice
description: >
  Pycaret을 사용하여 juice를 돌려봄
sitemap: false
---


```python
import torch
```


```python
torch.cuda.is_available()
```




    True




```python
from pycaret.datasets import get_data
dataset = get_data('juice')
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
      <th>Id</th>
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>Store7</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>STORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CH</td>
      <td>237</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500000</td>
      <td>1.99</td>
      <td>1.75</td>
      <td>0.24</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>CH</td>
      <td>239</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0.600000</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>CH</td>
      <td>245</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.17</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.680000</td>
      <td>2.09</td>
      <td>1.69</td>
      <td>0.40</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.091398</td>
      <td>0.23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>MM</td>
      <td>227</td>
      <td>1</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.400000</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>CH</td>
      <td>228</td>
      <td>7</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.956535</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
from pycaret.classification import *
setup_clf = setup(data=dataset, target='Purchase')
```


<style  type="text/css" >
#T_0a9cc58a_3ba2_11ed_aef1_38d547025041row44_col1{
            background-color:  lightgreen;
        }</style><table id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row0_col1" class="data row0 col1" >288</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row1_col0" class="data row1 col0" >Target</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row1_col1" class="data row1 col1" >Purchase</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row2_col0" class="data row2 col0" >Target Type</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row2_col1" class="data row2 col1" >Binary</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row3_col0" class="data row3 col0" >Label Encoded</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row3_col1" class="data row3 col1" >CH: 0, MM: 1</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row4_col0" class="data row4 col0" >Original Data</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row4_col1" class="data row4 col1" >(1070, 19)</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row5_col0" class="data row5 col0" >Missing Values</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row5_col1" class="data row5 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row6_col0" class="data row6 col0" >Numeric Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row6_col1" class="data row6 col1" >13</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row7_col0" class="data row7 col0" >Categorical Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row7_col1" class="data row7 col1" >5</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row8_col0" class="data row8 col0" >Ordinal Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row8_col1" class="data row8 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row9_col0" class="data row9 col0" >High Cardinality Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row9_col1" class="data row9 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row10_col0" class="data row10 col0" >High Cardinality Method</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row10_col1" class="data row10 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row11_col1" class="data row11 col1" >(748, 17)</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row12_col1" class="data row12 col1" >(322, 17)</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row13_col0" class="data row13 col0" >Shuffle Train-Test</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row13_col1" class="data row13 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row14_col0" class="data row14 col0" >Stratify Train-Test</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row14_col1" class="data row14 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row15_col0" class="data row15 col0" >Fold Generator</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row15_col1" class="data row15 col1" >StratifiedKFold</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row16_col0" class="data row16 col0" >Fold Number</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row16_col1" class="data row16 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row17_col0" class="data row17 col0" >CPU Jobs</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row17_col1" class="data row17 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row18_col0" class="data row18 col0" >Use GPU</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row18_col1" class="data row18 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row19_col0" class="data row19 col0" >Log Experiment</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row20_col0" class="data row20 col0" >Experiment Name</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row20_col1" class="data row20 col1" >clf-default-name</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row21_col0" class="data row21 col0" >USI</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row21_col1" class="data row21 col1" >ad07</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row22_col0" class="data row22 col0" >Imputation Type</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row22_col1" class="data row22 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row23_col0" class="data row23 col0" >Iterative Imputation Iteration</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row24_col0" class="data row24 col0" >Numeric Imputer</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row24_col1" class="data row24 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row25_col0" class="data row25 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row26_col0" class="data row26 col0" >Categorical Imputer</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row26_col1" class="data row26 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row27_col0" class="data row27 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row28_col0" class="data row28 col0" >Unknown Categoricals Handling</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row28_col1" class="data row28 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row29_col0" class="data row29 col0" >Normalize</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row30_col0" class="data row30 col0" >Normalize Method</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row31_col0" class="data row31 col0" >Transformation</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row32_col0" class="data row32 col0" >Transformation Method</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row33_col0" class="data row33 col0" >PCA</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row33_col1" class="data row33 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row34_col0" class="data row34 col0" >PCA Method</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row34_col1" class="data row34 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row35_col0" class="data row35 col0" >PCA Components</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row36_col0" class="data row36 col0" >Ignore Low Variance</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row37_col0" class="data row37 col0" >Combine Rare Levels</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row38_col0" class="data row38 col0" >Rare Level Threshold</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row39_col0" class="data row39 col0" >Numeric Binning</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row40_col0" class="data row40 col0" >Remove Outliers</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row41_col0" class="data row41 col0" >Outliers Threshold</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row42_col0" class="data row42 col0" >Remove Multicollinearity</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row43_col0" class="data row43 col0" >Multicollinearity Threshold</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row44_col0" class="data row44 col0" >Remove Perfect Collinearity</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row44_col1" class="data row44 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row45_col0" class="data row45 col0" >Clustering</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row45_col1" class="data row45 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row46_col0" class="data row46 col0" >Clustering Iteration</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row46_col1" class="data row46 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row47_col0" class="data row47 col0" >Polynomial Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row47_col1" class="data row47 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row48_col0" class="data row48 col0" >Polynomial Degree</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row48_col1" class="data row48 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row49_col0" class="data row49 col0" >Trignometry Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row49_col1" class="data row49 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row50_col0" class="data row50 col0" >Polynomial Threshold</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row50_col1" class="data row50 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row51_col0" class="data row51 col0" >Group Features</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row51_col1" class="data row51 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row52_col0" class="data row52 col0" >Feature Selection</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row52_col1" class="data row52 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row53_col0" class="data row53 col0" >Feature Selection Method</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row53_col1" class="data row53 col1" >classic</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row54_col0" class="data row54 col0" >Features Selection Threshold</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row54_col1" class="data row54 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row55_col0" class="data row55 col0" >Feature Interaction</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row55_col1" class="data row55 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row56_col0" class="data row56 col0" >Feature Ratio</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row56_col1" class="data row56 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row57" class="row_heading level0 row57" >57</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row57_col0" class="data row57 col0" >Interaction Threshold</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row57_col1" class="data row57 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row58" class="row_heading level0 row58" >58</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row58_col0" class="data row58 col0" >Fix Imbalance</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row58_col1" class="data row58 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041level0_row59" class="row_heading level0 row59" >59</th>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row59_col0" class="data row59 col0" >Fix Imbalance Method</td>
                        <td id="T_0a9cc58a_3ba2_11ed_aef1_38d547025041row59_col1" class="data row59 col1" >SMOTE</td>
            </tr>
    </tbody></table>



```python
rf = create_model('rf',fold=5)
```


<style  type="text/css" >
#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col0,#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col1,#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col2,#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col3,#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col4,#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col5,#T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col6{
            background:  yellow;
        }</style><table id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col0" class="data row0 col0" >0.8000</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col1" class="data row0 col1" >0.8649</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col2" class="data row0 col2" >0.7018</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col3" class="data row0 col3" >0.7547</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col4" class="data row0 col4" >0.7273</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col5" class="data row0 col5" >0.5697</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row0_col6" class="data row0 col6" >0.5706</td>
            </tr>
            <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col0" class="data row1 col0" >0.7933</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col1" class="data row1 col1" >0.8453</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col2" class="data row1 col2" >0.7895</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col3" class="data row1 col3" >0.7031</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col4" class="data row1 col4" >0.7438</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col5" class="data row1 col5" >0.5716</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row1_col6" class="data row1 col6" >0.5743</td>
            </tr>
            <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col0" class="data row2 col0" >0.8000</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col1" class="data row2 col1" >0.8817</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col2" class="data row2 col2" >0.7018</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col3" class="data row2 col3" >0.7547</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col4" class="data row2 col4" >0.7273</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col5" class="data row2 col5" >0.5697</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row2_col6" class="data row2 col6" >0.5706</td>
            </tr>
            <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col0" class="data row3 col0" >0.7248</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col1" class="data row3 col1" >0.8169</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col2" class="data row3 col2" >0.6429</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col3" class="data row3 col3" >0.6316</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col4" class="data row3 col4" >0.6372</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col5" class="data row3 col5" >0.4156</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row3_col6" class="data row3 col6" >0.4156</td>
            </tr>
            <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col0" class="data row4 col0" >0.8054</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col1" class="data row4 col1" >0.8380</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col2" class="data row4 col2" >0.7679</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col3" class="data row4 col3" >0.7288</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col4" class="data row4 col4" >0.7478</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col5" class="data row4 col5" >0.5895</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row4_col6" class="data row4 col6" >0.5901</td>
            </tr>
            <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col0" class="data row5 col0" >0.7847</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col1" class="data row5 col1" >0.8494</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col2" class="data row5 col2" >0.7207</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col3" class="data row5 col3" >0.7146</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col4" class="data row5 col4" >0.7167</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col5" class="data row5 col5" >0.5432</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row5_col6" class="data row5 col6" >0.5443</td>
            </tr>
            <tr>
                        <th id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041level0_row6" class="row_heading level0 row6" >Std</th>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col0" class="data row6 col0" >0.0302</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col1" class="data row6 col1" >0.0223</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col2" class="data row6 col2" >0.0524</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col3" class="data row6 col3" >0.0457</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col4" class="data row6 col4" >0.0406</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col5" class="data row6 col5" >0.0643</td>
                        <td id="T_2bb5aef8_3ba2_11ed_aef1_38d547025041row6_col6" class="data row6 col6" >0.0647</td>
            </tr>
    </tbody></table>



```python
top5 = compare_models(sort='Accuracy', n_select=5)
```


<style  type="text/css" >
    #T_4729695e_3ba2_11ed_aef1_38d547025041 th {
          text-align: left;
    }#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col3,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col0,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col7{
            text-align:  left;
            text-align:  left;
        }#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col4,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col1,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col5,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col6,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col7,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col2,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col3{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
        }#T_4729695e_3ba2_11ed_aef1_38d547025041row0_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row1_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row2_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row3_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row4_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row5_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row6_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row7_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row9_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row10_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row13_col8{
            text-align:  left;
            text-align:  left;
            background-color:  lightgrey;
        }#T_4729695e_3ba2_11ed_aef1_38d547025041row8_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row11_col8,#T_4729695e_3ba2_11ed_aef1_38d547025041row12_col8{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
            background-color:  lightgrey;
        }</style><table id="T_4729695e_3ba2_11ed_aef1_38d547025041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>        <th class="col_heading level0 col8" >TT (Sec)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row0" class="row_heading level0 row0" >lr</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col0" class="data row0 col0" >Logistic Regression</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col1" class="data row0 col1" >0.8155</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col2" class="data row0 col2" >0.8868</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col3" class="data row0 col3" >0.7277</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col4" class="data row0 col4" >0.7756</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col5" class="data row0 col5" >0.7491</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col6" class="data row0 col6" >0.6036</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col7" class="data row0 col7" >0.6061</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row0_col8" class="data row0 col8" >0.1880</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row1" class="row_heading level0 row1" >ridge</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col0" class="data row1 col0" >Ridge Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col1" class="data row1 col1" >0.8155</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col2" class="data row1 col2" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col3" class="data row1 col3" >0.7523</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col4" class="data row1 col4" >0.7627</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col5" class="data row1 col5" >0.7557</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col6" class="data row1 col6" >0.6077</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col7" class="data row1 col7" >0.6095</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row1_col8" class="data row1 col8" >0.0070</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row2" class="row_heading level0 row2" >lda</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col0" class="data row2 col0" >Linear Discriminant Analysis</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col1" class="data row2 col1" >0.8101</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col2" class="data row2 col2" >0.8876</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col3" class="data row2 col3" >0.7346</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col4" class="data row2 col4" >0.7596</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col5" class="data row2 col5" >0.7448</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col6" class="data row2 col6" >0.5940</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col7" class="data row2 col7" >0.5963</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row2_col8" class="data row2 col8" >0.0060</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row3" class="row_heading level0 row3" >gbc</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col0" class="data row3 col0" >Gradient Boosting Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col1" class="data row3 col1" >0.8007</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col2" class="data row3 col2" >0.8756</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col3" class="data row3 col3" >0.7203</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col4" class="data row3 col4" >0.7459</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col5" class="data row3 col5" >0.7301</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col6" class="data row3 col6" >0.5725</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col7" class="data row3 col7" >0.5755</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row3_col8" class="data row3 col8" >0.0230</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row4" class="row_heading level0 row4" >ada</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col0" class="data row4 col0" >Ada Boost Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col1" class="data row4 col1" >0.7968</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col2" class="data row4 col2" >0.8640</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col3" class="data row4 col3" >0.7060</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col4" class="data row4 col4" >0.7462</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col5" class="data row4 col5" >0.7228</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col6" class="data row4 col6" >0.5629</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col7" class="data row4 col7" >0.5661</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row4_col8" class="data row4 col8" >0.0210</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row5" class="row_heading level0 row5" >rf</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col0" class="data row5 col0" >Random Forest Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col1" class="data row5 col1" >0.7888</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col2" class="data row5 col2" >0.8553</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col3" class="data row5 col3" >0.7346</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col4" class="data row5 col4" >0.7189</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col5" class="data row5 col5" >0.7218</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col6" class="data row5 col6" >0.5523</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col7" class="data row5 col7" >0.5571</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row5_col8" class="data row5 col8" >0.1100</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row6" class="row_heading level0 row6" >lightgbm</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col0" class="data row6 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col1" class="data row6 col1" >0.7860</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col2" class="data row6 col2" >0.8620</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col3" class="data row6 col3" >0.7314</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col4" class="data row6 col4" >0.7149</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col5" class="data row6 col5" >0.7199</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col6" class="data row6 col6" >0.5473</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col7" class="data row6 col7" >0.5506</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row6_col8" class="data row6 col8" >0.0160</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row7" class="row_heading level0 row7" >et</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col0" class="data row7 col0" >Extra Trees Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col1" class="data row7 col1" >0.7821</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col2" class="data row7 col2" >0.8178</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col3" class="data row7 col3" >0.7347</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col4" class="data row7 col4" >0.7053</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col5" class="data row7 col5" >0.7147</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col6" class="data row7 col6" >0.5391</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col7" class="data row7 col7" >0.5447</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row7_col8" class="data row7 col8" >0.1090</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row8" class="row_heading level0 row8" >dt</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col0" class="data row8 col0" >Decision Tree Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col1" class="data row8 col1" >0.7540</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col2" class="data row8 col2" >0.7529</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col3" class="data row8 col3" >0.7031</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col4" class="data row8 col4" >0.6696</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col5" class="data row8 col5" >0.6821</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col6" class="data row8 col6" >0.4822</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col7" class="data row8 col7" >0.4863</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row8_col8" class="data row8 col8" >0.0050</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row9" class="row_heading level0 row9" >nb</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col0" class="data row9 col0" >Naive Bayes</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col1" class="data row9 col1" >0.7499</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col2" class="data row9 col2" >0.8248</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col3" class="data row9 col3" >0.7527</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col4" class="data row9 col4" >0.6475</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col5" class="data row9 col5" >0.6946</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col6" class="data row9 col6" >0.4854</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col7" class="data row9 col7" >0.4913</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row9_col8" class="data row9 col8" >0.0060</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row10" class="row_heading level0 row10" >knn</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col0" class="data row10 col0" >K Neighbors Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col1" class="data row10 col1" >0.7314</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col2" class="data row10 col2" >0.7596</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col3" class="data row10 col3" >0.5930</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col4" class="data row10 col4" >0.6646</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col5" class="data row10 col5" >0.6242</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col6" class="data row10 col6" >0.4166</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col7" class="data row10 col7" >0.4202</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row10_col8" class="data row10 col8" >0.0280</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row11" class="row_heading level0 row11" >dummy</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col0" class="data row11 col0" >Dummy Classifier</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col1" class="data row11 col1" >0.6217</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col2" class="data row11 col2" >0.5000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col3" class="data row11 col3" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col4" class="data row11 col4" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col5" class="data row11 col5" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col6" class="data row11 col6" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col7" class="data row11 col7" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row11_col8" class="data row11 col8" >0.0050</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row12" class="row_heading level0 row12" >svm</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col0" class="data row12 col0" >SVM - Linear Kernel</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col1" class="data row12 col1" >0.5240</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col2" class="data row12 col2" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col3" class="data row12 col3" >0.4000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col4" class="data row12 col4" >0.1512</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col5" class="data row12 col5" >0.2194</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col6" class="data row12 col6" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col7" class="data row12 col7" >0.0000</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row12_col8" class="data row12 col8" >0.0050</td>
            </tr>
            <tr>
                        <th id="T_4729695e_3ba2_11ed_aef1_38d547025041level0_row13" class="row_heading level0 row13" >qda</th>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col0" class="data row13 col0" >Quadratic Discriminant Analysis</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col1" class="data row13 col1" >0.3717</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col2" class="data row13 col2" >0.4771</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col3" class="data row13 col3" >0.9034</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col4" class="data row13 col4" >0.3591</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col5" class="data row13 col5" >0.5110</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col6" class="data row13 col6" >-0.0424</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col7" class="data row13 col7" >-0.0494</td>
                        <td id="T_4729695e_3ba2_11ed_aef1_38d547025041row13_col8" class="data row13 col8" >0.0060</td>
            </tr>
    </tbody></table>



```python
top5
```




    [LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None, max_iter=1000,
                        multi_class='auto', n_jobs=None, penalty='l2',
                        random_state=288, solver='lbfgs', tol=0.0001, verbose=0,
                        warm_start=False),
     RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                     max_iter=None, normalize=False, random_state=288, solver='auto',
                     tol=0.001),
     LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                solver='svd', store_covariance=False, tol=0.0001),
     GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                learning_rate=0.1, loss='deviance', max_depth=3,
                                max_features=None, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=100,
                                n_iter_no_change=None, presort='deprecated',
                                random_state=288, subsample=1.0, tol=0.0001,
                                validation_fraction=0.1, verbose=0,
                                warm_start=False),
     AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                        n_estimators=50, random_state=288)]




```python
tuned_top5 = [tune_model(i) for i in top5]
```


<style  type="text/css" >
#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col0,#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col1,#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col2,#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col3,#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col4,#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col5,#T_94c07554_3ba2_11ed_aef1_38d547025041row10_col6{
            background:  yellow;
        }</style><table id="T_94c07554_3ba2_11ed_aef1_38d547025041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col0" class="data row0 col0" >0.8267</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col1" class="data row0 col1" >0.9003</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col2" class="data row0 col2" >0.7586</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col3" class="data row0 col3" >0.7857</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col4" class="data row0 col4" >0.7719</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col5" class="data row0 col5" >0.6322</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row0_col6" class="data row0 col6" >0.6325</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col0" class="data row1 col0" >0.8667</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col1" class="data row1 col1" >0.9067</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col2" class="data row1 col2" >0.7241</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col3" class="data row1 col3" >0.9130</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col4" class="data row1 col4" >0.8077</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col5" class="data row1 col5" >0.7077</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row1_col6" class="data row1 col6" >0.7189</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col0" class="data row2 col0" >0.8533</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col1" class="data row2 col1" >0.9153</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col2" class="data row2 col2" >0.8276</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col3" class="data row2 col3" >0.8000</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col4" class="data row2 col4" >0.8136</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col5" class="data row2 col5" >0.6927</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row2_col6" class="data row2 col6" >0.6930</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col0" class="data row3 col0" >0.7867</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col1" class="data row3 col1" >0.8894</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col2" class="data row3 col2" >0.7500</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col3" class="data row3 col3" >0.7000</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col4" class="data row3 col4" >0.7241</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col5" class="data row3 col5" >0.5506</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row3_col6" class="data row3 col6" >0.5514</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col0" class="data row4 col0" >0.8667</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col1" class="data row4 col1" >0.9362</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col2" class="data row4 col2" >0.8929</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col3" class="data row4 col3" >0.7812</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col4" class="data row4 col4" >0.8333</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col5" class="data row4 col5" >0.7230</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row4_col6" class="data row4 col6" >0.7275</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col0" class="data row5 col0" >0.8400</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col1" class="data row5 col1" >0.9236</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col2" class="data row5 col2" >0.7500</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col3" class="data row5 col3" >0.8077</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col4" class="data row5 col4" >0.7778</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col5" class="data row5 col5" >0.6530</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row5_col6" class="data row5 col6" >0.6541</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col0" class="data row6 col0" >0.7467</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col1" class="data row6 col1" >0.8469</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col2" class="data row6 col2" >0.5714</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col3" class="data row6 col3" >0.6957</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col4" class="data row6 col4" >0.6275</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col5" class="data row6 col5" >0.4383</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row6_col6" class="data row6 col6" >0.4432</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col0" class="data row7 col0" >0.7600</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col1" class="data row7 col1" >0.8100</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col2" class="data row7 col2" >0.6429</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col3" class="data row7 col3" >0.6923</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col4" class="data row7 col4" >0.6667</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col5" class="data row7 col5" >0.4796</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row7_col6" class="data row7 col6" >0.4804</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col0" class="data row8 col0" >0.8649</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col1" class="data row8 col1" >0.8769</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col2" class="data row8 col2" >0.7500</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col3" class="data row8 col3" >0.8750</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col4" class="data row8 col4" >0.8077</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col5" class="data row8 col5" >0.7045</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row8_col6" class="data row8 col6" >0.7094</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col0" class="data row9 col0" >0.7568</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col1" class="data row9 col1" >0.8688</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col2" class="data row9 col2" >0.7143</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col3" class="data row9 col3" >0.6667</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col4" class="data row9 col4" >0.6897</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col5" class="data row9 col5" >0.4900</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row9_col6" class="data row9 col6" >0.4908</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col0" class="data row10 col0" >0.8168</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col1" class="data row10 col1" >0.8874</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col2" class="data row10 col2" >0.7382</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col3" class="data row10 col3" >0.7717</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col4" class="data row10 col4" >0.7520</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col5" class="data row10 col5" >0.6072</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row10_col6" class="data row10 col6" >0.6101</td>
            </tr>
            <tr>
                        <th id="T_94c07554_3ba2_11ed_aef1_38d547025041level0_row11" class="row_heading level0 row11" >Std</th>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col0" class="data row11 col0" >0.0468</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col1" class="data row11 col1" >0.0362</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col2" class="data row11 col2" >0.0839</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col3" class="data row11 col3" >0.0783</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col4" class="data row11 col4" >0.0672</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col5" class="data row11 col5" >0.1024</td>
                        <td id="T_94c07554_3ba2_11ed_aef1_38d547025041row11_col6" class="data row11 col6" >0.1035</td>
            </tr>
    </tbody></table>



```python
blender_top5 = blend_models(estimator_list=tuned_top5)
```


<style  type="text/css" >
#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col0,#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col1,#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col2,#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col3,#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col4,#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col5,#T_b0845558_3ba2_11ed_aef1_38d547025041row10_col6{
            background:  yellow;
        }</style><table id="T_b0845558_3ba2_11ed_aef1_38d547025041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col0" class="data row0 col0" >0.8667</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col1" class="data row0 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col2" class="data row0 col2" >0.8621</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col3" class="data row0 col3" >0.8065</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col4" class="data row0 col4" >0.8333</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col5" class="data row0 col5" >0.7224</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row0_col6" class="data row0 col6" >0.7235</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col0" class="data row1 col0" >0.8267</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col1" class="data row1 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col2" class="data row1 col2" >0.7586</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col3" class="data row1 col3" >0.7857</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col4" class="data row1 col4" >0.7719</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col5" class="data row1 col5" >0.6322</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row1_col6" class="data row1 col6" >0.6325</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col0" class="data row2 col0" >0.8667</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col1" class="data row2 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col2" class="data row2 col2" >0.8621</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col3" class="data row2 col3" >0.8065</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col4" class="data row2 col4" >0.8333</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col5" class="data row2 col5" >0.7224</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row2_col6" class="data row2 col6" >0.7235</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col0" class="data row3 col0" >0.7867</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col1" class="data row3 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col2" class="data row3 col2" >0.7857</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col3" class="data row3 col3" >0.6875</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col4" class="data row3 col4" >0.7333</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col5" class="data row3 col5" >0.5569</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row3_col6" class="data row3 col6" >0.5603</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col0" class="data row4 col0" >0.8800</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col1" class="data row4 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col2" class="data row4 col2" >0.8929</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col3" class="data row4 col3" >0.8065</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col4" class="data row4 col4" >0.8475</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col5" class="data row4 col5" >0.7490</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row4_col6" class="data row4 col6" >0.7516</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col0" class="data row5 col0" >0.8533</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col1" class="data row5 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col2" class="data row5 col2" >0.7857</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col3" class="data row5 col3" >0.8148</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col4" class="data row5 col4" >0.8000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col5" class="data row5 col5" >0.6843</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row5_col6" class="data row5 col6" >0.6846</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col0" class="data row6 col0" >0.8000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col1" class="data row6 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col2" class="data row6 col2" >0.7143</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col3" class="data row6 col3" >0.7407</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col4" class="data row6 col4" >0.7273</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col5" class="data row6 col5" >0.5695</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row6_col6" class="data row6 col6" >0.5697</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col0" class="data row7 col0" >0.7333</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col1" class="data row7 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col2" class="data row7 col2" >0.6786</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col3" class="data row7 col3" >0.6333</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col4" class="data row7 col4" >0.6552</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col5" class="data row7 col5" >0.4382</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row7_col6" class="data row7 col6" >0.4389</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col0" class="data row8 col0" >0.8514</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col1" class="data row8 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col2" class="data row8 col2" >0.7143</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col3" class="data row8 col3" >0.8696</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col4" class="data row8 col4" >0.7843</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col5" class="data row8 col5" >0.6726</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row8_col6" class="data row8 col6" >0.6801</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col0" class="data row9 col0" >0.7568</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col1" class="data row9 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col2" class="data row9 col2" >0.7143</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col3" class="data row9 col3" >0.6667</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col4" class="data row9 col4" >0.6897</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col5" class="data row9 col5" >0.4900</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row9_col6" class="data row9 col6" >0.4908</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col0" class="data row10 col0" >0.8221</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col1" class="data row10 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col2" class="data row10 col2" >0.7768</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col3" class="data row10 col3" >0.7618</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col4" class="data row10 col4" >0.7676</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col5" class="data row10 col5" >0.6237</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row10_col6" class="data row10 col6" >0.6256</td>
            </tr>
            <tr>
                        <th id="T_b0845558_3ba2_11ed_aef1_38d547025041level0_row11" class="row_heading level0 row11" >Std</th>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col0" class="data row11 col0" >0.0480</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col1" class="data row11 col1" >0.0000</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col2" class="data row11 col2" >0.0706</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col3" class="data row11 col3" >0.0725</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col4" class="data row11 col4" >0.0615</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col5" class="data row11 col5" >0.1005</td>
                        <td id="T_b0845558_3ba2_11ed_aef1_38d547025041row11_col6" class="data row11 col6" >0.1010</td>
            </tr>
    </tbody></table>



```python
final_model = finalize_model(blender_top5)
prediction = predict_model(final_model, data=dataset.iloc[-100:])
```


<style  type="text/css" >
</style><table id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col0" class="data row0 col0" >Voting Classifier</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col1" class="data row0 col1" >0</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col2" class="data row0 col2" >0.8000</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col3" class="data row0 col3" >0</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col4" class="data row0 col4" >0</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col5" class="data row0 col5" >0</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col6" class="data row0 col6" >0</td>
                        <td id="T_e4b2e4d4_3ba2_11ed_aef1_38d547025041row0_col7" class="data row0 col7" >0</td>
            </tr>
    </tbody></table>



```python
prediction
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
      <th>Id</th>
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>Store7</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>STORE</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>970</th>
      <td>971</td>
      <td>MM</td>
      <td>240</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.0</td>
      <td>0.30</td>
      <td>0</td>
      <td>1</td>
      <td>0.224526</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
      <td>MM</td>
    </tr>
    <tr>
      <th>971</th>
      <td>972</td>
      <td>MM</td>
      <td>241</td>
      <td>1</td>
      <td>1.86</td>
      <td>1.99</td>
      <td>0.0</td>
      <td>0.30</td>
      <td>0</td>
      <td>1</td>
      <td>0.179621</td>
      <td>1.69</td>
      <td>1.86</td>
      <td>-0.17</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.13</td>
      <td>1</td>
      <td>MM</td>
    </tr>
    <tr>
      <th>972</th>
      <td>973</td>
      <td>CH</td>
      <td>242</td>
      <td>1</td>
      <td>1.86</td>
      <td>1.99</td>
      <td>0.0</td>
      <td>0.30</td>
      <td>0</td>
      <td>1</td>
      <td>0.143697</td>
      <td>1.69</td>
      <td>1.86</td>
      <td>-0.17</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.13</td>
      <td>1</td>
      <td>MM</td>
    </tr>
    <tr>
      <th>973</th>
      <td>974</td>
      <td>MM</td>
      <td>243</td>
      <td>1</td>
      <td>1.86</td>
      <td>1.99</td>
      <td>0.0</td>
      <td>0.80</td>
      <td>0</td>
      <td>1</td>
      <td>0.314957</td>
      <td>1.19</td>
      <td>1.86</td>
      <td>-0.67</td>
      <td>No</td>
      <td>0.402010</td>
      <td>0.000000</td>
      <td>0.13</td>
      <td>1</td>
      <td>MM</td>
    </tr>
    <tr>
      <th>974</th>
      <td>975</td>
      <td>MM</td>
      <td>244</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.251966</td>
      <td>2.09</td>
      <td>1.86</td>
      <td>0.23</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.23</td>
      <td>1</td>
      <td>MM</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1065</th>
      <td>1066</td>
      <td>CH</td>
      <td>252</td>
      <td>7</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.1</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.587822</td>
      <td>2.09</td>
      <td>1.76</td>
      <td>0.33</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.053763</td>
      <td>0.23</td>
      <td>0</td>
      <td>CH</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>1067</td>
      <td>CH</td>
      <td>256</td>
      <td>7</td>
      <td>1.86</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.670258</td>
      <td>2.18</td>
      <td>1.86</td>
      <td>0.32</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.32</td>
      <td>0</td>
      <td>CH</td>
    </tr>
    <tr>
      <th>1067</th>
      <td>1068</td>
      <td>MM</td>
      <td>257</td>
      <td>7</td>
      <td>1.86</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.736206</td>
      <td>2.18</td>
      <td>1.86</td>
      <td>0.32</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.32</td>
      <td>0</td>
      <td>CH</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>1069</td>
      <td>CH</td>
      <td>261</td>
      <td>7</td>
      <td>1.86</td>
      <td>2.13</td>
      <td>0.0</td>
      <td>0.24</td>
      <td>0</td>
      <td>0</td>
      <td>0.588965</td>
      <td>1.89</td>
      <td>1.86</td>
      <td>0.03</td>
      <td>Yes</td>
      <td>0.112676</td>
      <td>0.000000</td>
      <td>0.27</td>
      <td>0</td>
      <td>CH</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>1070</td>
      <td>CH</td>
      <td>270</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0.671172</td>
      <td>2.18</td>
      <td>1.86</td>
      <td>0.32</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.32</td>
      <td>1</td>
      <td>CH</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 20 columns</p>
</div>




```python
from pycaret.utils import check_metric
check_metric(prediction['Purchase'],prediction['Label'],metric='Accuracy')
```




    0.81




```python

```
