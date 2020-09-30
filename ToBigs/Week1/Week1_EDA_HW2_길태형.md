```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('seaborn')
warnings.filterwarnings('ignore')
%matplotlib inline
```

# 데이터 input & 대략적인 데이터 확인


```python
# 경로 설정 (노트북 파일 기점으로 상대경로 혹은 절대경로)
org_data = pd.read_csv('Auction_master_train.csv')
data = org_data
#data = pd.read_csv(r'C:\Users\Ki_Yoon_Yoo\Desktop\ToBigs/train.csv')

# Display first five rows 
data.head(20)
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
      <th>Auction_key</th>
      <th>Auction_class</th>
      <th>Bid_class</th>
      <th>Claim_price</th>
      <th>Appraisal_company</th>
      <th>Appraisal_date</th>
      <th>Auction_count</th>
      <th>Auction_miscarriage_count</th>
      <th>Total_land_gross_area</th>
      <th>Total_land_real_area</th>
      <th>...</th>
      <th>Specific</th>
      <th>Share_auction_YorN</th>
      <th>road_name</th>
      <th>road_bunji1</th>
      <th>road_bunji2</th>
      <th>Close_date</th>
      <th>Close_result</th>
      <th>point.y</th>
      <th>point.x</th>
      <th>Hammer_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2687</td>
      <td>임의</td>
      <td>개별</td>
      <td>1766037301</td>
      <td>정명감정</td>
      <td>2017-07-26 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>12592.0</td>
      <td>37.35</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>해운대해변로</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>2018-06-14 00:00:00</td>
      <td>배당</td>
      <td>35.162717</td>
      <td>129.137048</td>
      <td>760000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2577</td>
      <td>임의</td>
      <td>일반</td>
      <td>152946867</td>
      <td>희감정</td>
      <td>2016-09-12 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>42478.1</td>
      <td>18.76</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>마린시티2로</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>2017-03-30 00:00:00</td>
      <td>배당</td>
      <td>35.156633</td>
      <td>129.145068</td>
      <td>971889999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2197</td>
      <td>임의</td>
      <td>개별</td>
      <td>11326510</td>
      <td>혜림감정</td>
      <td>2016-11-22 00:00:00</td>
      <td>3</td>
      <td>2</td>
      <td>149683.1</td>
      <td>71.00</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>모라로110번길</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>2017-12-13 00:00:00</td>
      <td>배당</td>
      <td>35.184601</td>
      <td>128.996765</td>
      <td>93399999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2642</td>
      <td>임의</td>
      <td>일반</td>
      <td>183581724</td>
      <td>신라감정</td>
      <td>2016-12-13 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>24405.0</td>
      <td>32.98</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>황령대로319번가길</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>2017-12-27 00:00:00</td>
      <td>배당</td>
      <td>35.154180</td>
      <td>129.089081</td>
      <td>256899000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1958</td>
      <td>강제</td>
      <td>일반</td>
      <td>45887671</td>
      <td>나라감정</td>
      <td>2016-03-07 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>774.0</td>
      <td>45.18</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>오작로</td>
      <td>51.0</td>
      <td>NaN</td>
      <td>2016-10-04 00:00:00</td>
      <td>배당</td>
      <td>35.099630</td>
      <td>128.998874</td>
      <td>158660000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2238</td>
      <td>강제</td>
      <td>일반</td>
      <td>105437195</td>
      <td>한마음감정</td>
      <td>2017-01-03 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>8635.0</td>
      <td>41.39</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>일산봉로</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>2017-10-25 00:00:00</td>
      <td>배당</td>
      <td>35.086933</td>
      <td>129.065706</td>
      <td>206989000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1923</td>
      <td>임의</td>
      <td>일반</td>
      <td>137548730</td>
      <td>미래새한감정</td>
      <td>2016-01-19 00:00:00</td>
      <td>3</td>
      <td>2</td>
      <td>7927.0</td>
      <td>81.77</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>동삼서로</td>
      <td>61.0</td>
      <td>NaN</td>
      <td>2016-11-03 00:00:00</td>
      <td>배당</td>
      <td>35.084049</td>
      <td>129.070231</td>
      <td>135500000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2527</td>
      <td>임의</td>
      <td>일반</td>
      <td>506916971</td>
      <td>부일감정</td>
      <td>2016-04-28 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>331281.0</td>
      <td>92.22</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>분포로</td>
      <td>111.0</td>
      <td>NaN</td>
      <td>2016-11-22 00:00:00</td>
      <td>배당</td>
      <td>35.127808</td>
      <td>129.112206</td>
      <td>640299999</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2259</td>
      <td>강제</td>
      <td>일반</td>
      <td>40782876</td>
      <td>금정감정</td>
      <td>2016-03-08 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>1017.0</td>
      <td>66.44</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>천마로27번길</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>2016-09-27 00:00:00</td>
      <td>배당</td>
      <td>35.080817</td>
      <td>129.020155</td>
      <td>77380000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2666</td>
      <td>임의</td>
      <td>일반</td>
      <td>150000000</td>
      <td>연산감정</td>
      <td>2017-03-27 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>27447.2</td>
      <td>78.52</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>재반로84번길</td>
      <td>96.0</td>
      <td>7.0</td>
      <td>2017-12-27 00:00:00</td>
      <td>배당</td>
      <td>35.187903</td>
      <td>129.130913</td>
      <td>177070000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2362</td>
      <td>임의</td>
      <td>개별</td>
      <td>500000000</td>
      <td>명장감정</td>
      <td>2017-02-22 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>1205.8</td>
      <td>18.27</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>동평로405번길</td>
      <td>69.0</td>
      <td>NaN</td>
      <td>2018-02-28 00:00:00</td>
      <td>배당</td>
      <td>35.176581</td>
      <td>129.071228</td>
      <td>240000001</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1947</td>
      <td>임의</td>
      <td>일반</td>
      <td>37122950</td>
      <td>명장감정</td>
      <td>2016-02-18 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>5043.0</td>
      <td>37.38</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>태종로</td>
      <td>705.0</td>
      <td>NaN</td>
      <td>2017-02-13 00:00:00</td>
      <td>배당</td>
      <td>35.071897</td>
      <td>129.076585</td>
      <td>221017200</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2298</td>
      <td>임의</td>
      <td>일반</td>
      <td>25000000</td>
      <td>문일감정</td>
      <td>2016-09-26 00:00:00</td>
      <td>4</td>
      <td>2</td>
      <td>8880.0</td>
      <td>31.69</td>
      <td>...</td>
      <td>공유자 박수미,박종학으로부터 공유자 우선매수신고 있음.공유자가 민사집행법 제140조...</td>
      <td>Y</td>
      <td>비봉로</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>2018-04-19 00:00:00</td>
      <td>배당</td>
      <td>35.098222</td>
      <td>128.965136</td>
      <td>31888000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2550</td>
      <td>임의</td>
      <td>일반</td>
      <td>135000000</td>
      <td>미르감정</td>
      <td>2016-07-24 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>좌동순환로433번길</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>2017-07-24 00:00:00</td>
      <td>배당</td>
      <td>35.161948</td>
      <td>129.179110</td>
      <td>518800000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2046</td>
      <td>임의</td>
      <td>일반</td>
      <td>953680000</td>
      <td>국제감정</td>
      <td>2016-06-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>82385.1</td>
      <td>80.62</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>과정로343번길</td>
      <td>43.0</td>
      <td>NaN</td>
      <td>2016-12-20 00:00:00</td>
      <td>배당</td>
      <td>35.188580</td>
      <td>129.089335</td>
      <td>670190000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2678</td>
      <td>강제</td>
      <td>일반</td>
      <td>13848484</td>
      <td>드림감정</td>
      <td>2017-06-14 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>41235.0</td>
      <td>53.84</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>광안해변로</td>
      <td>418.0</td>
      <td>NaN</td>
      <td>2018-01-31 00:00:00</td>
      <td>배당</td>
      <td>35.159833</td>
      <td>129.131154</td>
      <td>531100000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2682</td>
      <td>임의</td>
      <td>일반</td>
      <td>149393725</td>
      <td>금정감정</td>
      <td>2017-07-04 00:00:00</td>
      <td>3</td>
      <td>2</td>
      <td>12133.7</td>
      <td>25.70</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>재반로</td>
      <td>141.0</td>
      <td>NaN</td>
      <td>2018-05-16 00:00:00</td>
      <td>배당</td>
      <td>35.191666</td>
      <td>129.126864</td>
      <td>143999000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1861</td>
      <td>임의</td>
      <td>개별</td>
      <td>1046685025</td>
      <td>대일감정</td>
      <td>2015-12-02 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>331.0</td>
      <td>7.59</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>차밭골로</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>2016-12-22 00:00:00</td>
      <td>배당</td>
      <td>35.216645</td>
      <td>129.079149</td>
      <td>133000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2037</td>
      <td>임의</td>
      <td>일반</td>
      <td>42479480</td>
      <td>미래새한감정</td>
      <td>2016-05-25 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>798.0</td>
      <td>41.69</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>웃서발로31번길</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2016-12-01 00:00:00</td>
      <td>배당</td>
      <td>35.081003</td>
      <td>129.068091</td>
      <td>126651100</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2103</td>
      <td>임의</td>
      <td>일반</td>
      <td>30665922</td>
      <td>한마음감정</td>
      <td>2016-08-02 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>1389.0</td>
      <td>39.43</td>
      <td>...</td>
      <td>NaN</td>
      <td>N</td>
      <td>수성로</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2017-04-05 00:00:00</td>
      <td>배당</td>
      <td>35.132840</td>
      <td>129.042040</td>
      <td>73899900</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 41 columns</p>
</div>




```python
print(data.describe())
```

           Auction_key   Claim_price  Auction_count  Auction_miscarriage_count  \
    count  1933.000000  1.933000e+03    1933.000000                1933.000000   
    mean   1380.271081  3.703908e+08       1.836006                   0.788412   
    std     801.670470  1.337869e+09       0.938319                   0.831715   
    min       1.000000  0.000000e+00       1.000000                   0.000000   
    25%     691.000000  7.746112e+07       1.000000                   0.000000   
    50%    1395.000000  1.728143e+08       2.000000                   1.000000   
    75%    2062.000000  3.565089e+08       2.000000                   1.000000   
    max    2762.000000  2.286481e+10      13.000000                   9.000000   
    
           Total_land_gross_area  Total_land_real_area  Total_land_auction_area  \
    count           1.933000e+03           1933.000000              1933.000000   
    mean            3.458714e+04             42.333802                41.310776   
    std             9.442101e+04             65.274404                65.385900   
    min             0.000000e+00              0.000000                 0.000000   
    25%             2.997000e+03             25.870000                24.570000   
    50%             1.424140e+04             37.510000                36.790000   
    75%             4.140310e+04             51.790000                51.320000   
    max             3.511936e+06           2665.840000              2665.840000   
    
           Total_building_area  Total_building_auction_area  \
    count          1933.000000                  1933.000000   
    mean             96.417693                    94.148810   
    std             106.323240                   106.845985   
    min               9.390000                     1.500000   
    25%              61.520000                    59.970000   
    50%              84.900000                    84.860000   
    75%             114.940000                   114.850000   
    max            4255.070000                  4255.070000   
    
           Total_appraisal_price  Minimum_sales_price  addr_bunji1  addr_bunji2  \
    count           1.933000e+03         1.933000e+03  1929.000000   889.000000   
    mean            4.973592e+08         4.155955e+08   601.952307    22.742407   
    std             7.873851e+08         5.030312e+08   554.119824    67.000807   
    min             4.285000e+06         4.285000e+06     1.000000     1.000000   
    25%             2.090000e+08         1.750000e+08   189.000000     1.000000   
    50%             3.600000e+08         3.120000e+08   482.000000     5.000000   
    75%             5.720000e+08         4.864000e+08   834.000000    18.000000   
    max             2.777500e+10         1.422080e+10  4937.000000  1414.000000   
    
           Total_floor  Current_floor  road_bunji1  road_bunji2      point.y  \
    count  1933.000000    1933.000000  1909.000000   155.000000  1933.000000   
    mean     16.980859       8.817900   127.441069    12.748387    36.698018   
    std       9.509021       8.044644   188.394217    10.735663     1.150269   
    min       3.000000       0.000000     1.000000     1.000000    35.051385   
    25%      12.000000       3.000000    24.000000     5.000000    35.188590   
    50%      15.000000       7.000000    57.000000     9.000000    37.500862   
    75%      21.000000      12.000000   145.000000    17.500000    37.566116   
    max      80.000000      65.000000  1716.000000    55.000000    37.685575   
    
               point.x  Hammer_price  
    count  1933.000000  1.933000e+03  
    mean    127.731667  4.726901e+08  
    std       0.993055  5.574493e+08  
    min     126.809393  6.303000e+06  
    25%     126.959167  1.975550e+08  
    50%     127.065003  3.544500e+08  
    75%     129.018054  5.599000e+08  
    max     129.255872  1.515100e+10  



```python
data.loc[data['Claim_price']<10000000,['Claim_price','Hammer_price']].describe()
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
      <th>Claim_price</th>
      <th>Hammer_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.100000e+01</td>
      <td>7.100000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.850455e+06</td>
      <td>3.758701e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.303591e+06</td>
      <td>4.230191e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>6.303000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000e+01</td>
      <td>1.056100e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.775241e+06</td>
      <td>2.639500e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.746602e+06</td>
      <td>4.453944e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.600000e+06</td>
      <td>2.086600e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
print('Types of Columns: ')
print(data.columns.values)
```

    Types of Columns: 
    ['Auction_key' 'Auction_class' 'Bid_class' 'Claim_price'
     'Appraisal_company' 'Appraisal_date' 'Auction_count'
     'Auction_miscarriage_count' 'Total_land_gross_area'
     'Total_land_real_area' 'Total_land_auction_area' 'Total_building_area'
     'Total_building_auction_area' 'Total_appraisal_price'
     'Minimum_sales_price' 'First_auction_date' 'Final_auction_date'
     'Final_result' 'Creditor' 'addr_do' 'addr_si' 'addr_dong' 'addr_li'
     'addr_san' 'addr_bunji1' 'addr_bunji2' 'addr_etc' 'Apartment_usage'
     'Preserve_regist_date' 'Total_floor' 'Current_floor' 'Specific'
     'Share_auction_YorN' 'road_name' 'road_bunji1' 'road_bunji2' 'Close_date'
     'Close_result' 'point.y' 'point.x' 'Hammer_price']



```python
print(data.dtypes)

```

    Auction_key                      int64
    Auction_class                   object
    Bid_class                       object
    Claim_price                      int64
    Appraisal_company               object
    Appraisal_date                  object
    Auction_count                    int64
    Auction_miscarriage_count        int64
    Total_land_gross_area          float64
    Total_land_real_area           float64
    Total_land_auction_area        float64
    Total_building_area            float64
    Total_building_auction_area    float64
    Total_appraisal_price            int64
    Minimum_sales_price              int64
    First_auction_date              object
    Final_auction_date              object
    Final_result                    object
    Creditor                        object
    addr_do                         object
    addr_si                         object
    addr_dong                       object
    addr_li                         object
    addr_san                        object
    addr_bunji1                    float64
    addr_bunji2                    float64
    addr_etc                        object
    Apartment_usage                 object
    Preserve_regist_date            object
    Total_floor                      int64
    Current_floor                    int64
    Specific                        object
    Share_auction_YorN              object
    road_name                       object
    road_bunji1                    float64
    road_bunji2                    float64
    Close_date                      object
    Close_result                    object
    point.y                        float64
    point.x                        float64
    Hammer_price                     int64
    dtype: object


# 과제 1 _결측치 찾기 & 처리하기


```python
print(data.isnull().sum())
```

    Auction_key                       0
    Auction_class                     0
    Bid_class                         0
    Claim_price                       0
    Appraisal_company                 0
    Appraisal_date                    0
    Auction_count                     0
    Auction_miscarriage_count         0
    Total_land_gross_area             0
    Total_land_real_area              0
    Total_land_auction_area           0
    Total_building_area               0
    Total_building_auction_area       0
    Total_appraisal_price             0
    Minimum_sales_price               0
    First_auction_date                0
    Final_auction_date                0
    Final_result                      0
    Creditor                          0
    addr_do                           0
    addr_si                           0
    addr_dong                         0
    addr_li                        1910
    addr_san                          0
    addr_bunji1                       4
    addr_bunji2                    1044
    addr_etc                          0
    Apartment_usage                   0
    Preserve_regist_date              0
    Total_floor                       0
    Current_floor                     0
    Specific                       1869
    Share_auction_YorN                0
    road_name                         0
    road_bunji1                      24
    road_bunji2                    1778
    Close_date                        0
    Close_result                      0
    point.y                           0
    point.x                           0
    Hammer_price                      0
    dtype: int64

결측치가 존재하는 항목중, 가장 쓸만하다고 판단되는 road_bunji를 처리하기로 했습니다

```python
print(data['road_bunji1'].unique())
```

    [3.000e+01 3.300e+01 8.800e+01 1.100e+02 5.100e+01 5.800e+01 6.100e+01
     1.110e+02 1.000e+01 9.600e+01 6.900e+01 7.050e+02 3.700e+01 4.300e+01
     4.180e+02 1.410e+02 2.400e+01 2.200e+01 2.000e+00 9.400e+01 3.130e+02
     2.100e+01 1.500e+01 2.330e+02 6.500e+01       nan 4.000e+01 1.580e+02
     1.030e+02 8.300e+01 4.500e+01 2.650e+02 2.300e+01 4.800e+01 1.490e+02
     1.700e+01 8.000e+00 1.040e+02 3.500e+02 5.040e+02 7.400e+01 1.003e+03
     9.100e+01 1.600e+01 3.400e+01 1.800e+01 2.680e+02 1.420e+02 5.600e+01
     6.200e+01 7.500e+01 5.690e+02 1.740e+02 2.600e+01 1.020e+02 3.390e+02
     2.900e+01 2.110e+02 1.100e+01 7.200e+01 2.700e+01 1.300e+01 4.200e+01
     1.200e+01 5.000e+01 4.480e+02 2.250e+02 1.000e+02 2.740e+02 1.550e+02
     6.170e+02 9.500e+01 1.000e+00 9.300e+01 8.700e+01 1.900e+01 3.200e+01
     2.000e+01 3.140e+02 8.200e+01 2.170e+02 7.100e+01 1.360e+02 7.000e+00
     2.940e+02 1.200e+02 5.270e+02 3.720e+02 3.800e+01 3.600e+01 8.400e+01
     6.000e+00 5.460e+02 1.090e+02 8.500e+01 9.800e+01 7.020e+02 3.100e+01
     4.320e+02 7.140e+02 4.700e+01 1.240e+02 3.650e+02 4.090e+02 5.000e+00
     1.620e+02 1.800e+02 1.700e+02 4.730e+02 2.750e+02 6.400e+01 1.750e+02
     6.000e+01 1.160e+02 1.530e+02 4.100e+01 7.000e+01 8.000e+01 2.000e+02
     5.300e+01 4.520e+02 2.310e+02 2.800e+01 4.600e+01 2.700e+02 6.800e+01
     8.100e+01 1.560e+02 1.080e+02 1.450e+02 1.940e+02 1.820e+02 1.670e+02
     1.650e+02 7.080e+02 1.380e+02 1.350e+02 8.900e+01 5.080e+02 9.700e+01
     3.500e+01 1.900e+02 1.480e+02 1.060e+02 3.430e+02 1.130e+02 1.850e+02
     1.280e+02 6.700e+01 5.900e+01 8.200e+02 2.500e+01 2.280e+02 4.590e+02
     2.520e+02 1.520e+02 1.180e+02 9.000e+00 4.900e+01 5.700e+01 9.900e+01
     1.660e+02 2.710e+02 8.600e+01 5.200e+01 1.890e+02 1.680e+02 1.600e+02
     5.500e+01 3.010e+02 1.270e+02 4.870e+02 7.600e+01 2.460e+02 1.400e+02
     7.120e+02 4.390e+02 5.420e+02 2.660e+02 4.000e+02 2.270e+02 3.900e+01
     7.540e+02 3.940e+02 7.300e+01 4.830e+02 1.220e+02 5.400e+01 3.000e+00
     4.770e+02 7.900e+01 1.400e+01 1.770e+02 5.020e+02 5.640e+02 1.330e+02
     3.660e+02 2.840e+02 5.840e+02 1.260e+02 3.360e+02 7.480e+02 1.300e+02
     2.360e+02 7.930e+02 3.000e+02 3.510e+02 2.320e+02 9.200e+01 6.840e+02
     2.180e+02 5.320e+02 1.630e+02 1.950e+02 7.520e+02 1.070e+02 3.030e+02
     2.300e+02 4.060e+02 8.130e+02 1.460e+02 3.480e+02 1.136e+03 1.050e+02
     2.130e+02 1.640e+02 4.000e+00 6.300e+01 2.640e+02 4.800e+02 6.090e+02
     2.240e+02 3.050e+02 1.706e+03 3.190e+02 7.700e+01 3.850e+02 3.070e+02
     3.100e+02 7.800e+01 9.320e+02 2.120e+02 1.720e+02 3.440e+02 1.920e+02
     2.210e+02 1.510e+02 3.820e+02 2.830e+02 7.800e+02 2.900e+02 8.590e+02
     6.460e+02 1.590e+02 2.340e+02 1.430e+02 1.760e+02 4.050e+02 2.200e+02
     2.230e+02 3.970e+02 4.350e+02 3.880e+02 5.750e+02 2.290e+02 1.540e+02
     8.160e+02 1.456e+03 1.910e+02 1.430e+03 2.500e+02 5.550e+02 6.370e+02
     1.830e+02 2.560e+02 6.160e+02 2.860e+02 4.640e+02 4.680e+02 2.350e+02
     1.240e+03 4.560e+02 8.450e+02 3.450e+02 5.300e+02 7.350e+02 1.880e+02
     1.190e+02 6.400e+02 3.990e+02 6.380e+02 5.140e+02 3.770e+02 1.500e+02
     9.000e+01 3.470e+02 9.130e+02 6.700e+02 2.010e+02 2.850e+02 1.470e+02
     8.510e+02 4.470e+02 5.790e+02 2.420e+02 1.290e+02 1.310e+02 8.560e+02
     5.970e+02 1.960e+02 2.980e+02 3.110e+02 1.930e+02 1.870e+02 1.044e+03
     1.370e+02 4.150e+02 2.070e+02 2.690e+02 2.430e+02 2.100e+02 3.530e+02
     1.150e+02 4.790e+02 4.170e+02 1.389e+03 5.950e+02 4.030e+02 3.210e+02
     5.680e+02 1.140e+02 1.980e+02 3.750e+02 2.570e+02 3.120e+02 1.716e+03
     1.152e+03 1.010e+02 3.710e+02 6.600e+01 1.250e+02 5.010e+02 2.990e+02
     5.520e+02 1.210e+02 1.320e+02 2.480e+02 3.670e+02 1.390e+02 1.340e+02
     1.340e+03 5.170e+02 5.890e+02 3.830e+02 6.040e+02 3.840e+02 5.760e+02
     3.200e+02 1.120e+03 4.280e+02 1.840e+02 1.218e+03 1.860e+02 3.280e+02]


## 결측치 처리

nan값을 찾아서 0으로 변경. 
road_bunji의 type이 float이기 때문에, string이 아닌 실수 값으로 저장.
road_bunji는 번지수 이기 때문에, 0은 사실상 null과 논리상으로 동일하다고 판단했습니다.


```python
data['road_bunji1'].fillna(0,inplace=True)
```


```python
print(data['road_bunji1'].astype(int).unique())
```

    [  30   33   88  110   51   58   61  111   10   96   69  705   37   43
      418  141   24   22    2   94  313   21   15  233   65    0   40  158
      103   83   45  265   23   48  149   17    8  104  350  504   74 1003
       91   16   34   18  268  142   56   62   75  569  174   26  102  339
       29  211   11   72   27   13   42   12   50  448  225  100  274  155
      617   95    1   93   87   19   32   20  314   82  217   71  136    7
      294  120  527  372   38   36   84    6  546  109   85   98  702   31
      432  714   47  124  365  409    5  162  180  170  473  275   64  175
       60  116  153   41   70   80  200   53  452  231   28   46  270   68
       81  156  108  145  194  182  167  165  708  138  135   89  508   97
       35  190  148  106  343  113  185  128   67   59  820   25  228  459
      252  152  118    9   49   57   99  166  271   86   52  189  168  160
       55  301  127  487   76  246  140  712  439  542  266  400  227   39
      754  394   73  483  122   54    3  477   79   14  177  502  564  133
      366  284  584  126  336  748  130  236  793  300  351  232   92  684
      218  532  163  195  752  107  303  230  406  813  146  348 1136  105
      213  164    4   63  264  480  609  224  305 1706  319   77  385  307
      310   78  932  212  172  344  192  221  151  382  283  780  290  859
      646  159  234  143  176  405  220  223  397  435  388  575  229  154
      816 1456  191 1430  250  555  637  183  256  616  286  464  468  235
     1240  456  845  345  530  735  188  119  640  399  638  514  377  150
       90  347  913  670  201  285  147  851  447  579  242  129  131  856
      597  196  298  311  193  187 1044  137  415  207  269  243  210  353
      115  479  417 1389  595  403  321  568  114  198  375  257  312 1716
     1152  101  371   66  125  501  299  552  121  132  248  367  139  134
     1340  517  589  383  604  384  576  320 1120  428  184 1218  186  328]


### road_bunji1 항목에 nan값이 얼마나 많은지 시각적으로 표현


```python
data.loc[data['road_bunji1']==0,'road_bunji1_Null'] = 'Null'
data.loc[data['road_bunji1']!=0,'road_bunji1_Null'] = 'Non-Null'
```


```python
data[['road_bunji1_Null','Auction_key']].groupby(['road_bunji1_Null']).count()
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
      <th>Auction_key</th>
    </tr>
    <tr>
      <th>road_bunji1_Null</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Non-Null</th>
      <td>1909</td>
    </tr>
    <tr>
      <th>Null</th>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[['road_bunji1_Null','Auction_key']].groupby(['road_bunji1_Null']).count().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x59b5bd0>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_19_1.png)


# 과제 2 _범주형 변수들 처리

## nominal 변수

- <경매 종류>
Auction_class _ 경매 구분 
Bid_class _ 입찰 구분
- <건물 위치>
addr_do _ 주소_시도
addr_si _ 주소_시군구
addr_dong _ 주소_읍면동
- <기타>
Apartment_usage _ 건물의 대표 용도
Share_auction_YorN _ 지분 경매 여부
Final_result _ 최종 결과 

- <경매종류>와 <기타>에 속한 변수들의 관계를 파악해보려 했지만, 밝히지 못했습니다. 

- <건물위치>에 속한 변수들은 관계가 있는것이 자명합니다.

- 예를 들어 한 건물의 addr_dong의 값이 '이태원동'이라면, 
addr_do는 '서울' 이고, addr_si는 '용산구' 일 수 밖에 없겠죠. 

- 따라서, 기타에 속한 변수들을 제외하고, 경매 종류에 속한 변수들만 encoding을 하고,
건물의 위치를 대략적으로 파악할 수 있도록, column으로 부산, 서울, 강남, 강북을 추가하고, one-hot encoding으로 값을 넣으려 합니다.

Auction_class 처리 


```python
print(data['Auction_class'].unique())
print(data['Bid_class'].unique())
print(data['Final_result'].unique())
```

    ['임의' '강제']
    ['개별' '일반' '일괄']
    ['낙찰']



```python
dummy_var = pd.get_dummies(data.Auction_class)
pd.concat([data.drop(['Auction_class'], axis=1),dummy_var], axis=1).head()
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
      <th>Auction_key</th>
      <th>Bid_class</th>
      <th>Claim_price</th>
      <th>Appraisal_company</th>
      <th>Appraisal_date</th>
      <th>Auction_count</th>
      <th>Auction_miscarriage_count</th>
      <th>Total_land_gross_area</th>
      <th>Total_land_real_area</th>
      <th>Total_land_auction_area</th>
      <th>...</th>
      <th>road_bunji1</th>
      <th>road_bunji2</th>
      <th>Close_date</th>
      <th>Close_result</th>
      <th>point.y</th>
      <th>point.x</th>
      <th>Hammer_price</th>
      <th>road_bunji1_Null</th>
      <th>강제</th>
      <th>임의</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2687</td>
      <td>개별</td>
      <td>1766037301</td>
      <td>정명감정</td>
      <td>2017-07-26 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>12592.0</td>
      <td>37.35</td>
      <td>37.35</td>
      <td>...</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>2018-06-14 00:00:00</td>
      <td>배당</td>
      <td>35.162717</td>
      <td>129.137048</td>
      <td>760000000</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2577</td>
      <td>일반</td>
      <td>152946867</td>
      <td>희감정</td>
      <td>2016-09-12 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>42478.1</td>
      <td>18.76</td>
      <td>18.76</td>
      <td>...</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>2017-03-30 00:00:00</td>
      <td>배당</td>
      <td>35.156633</td>
      <td>129.145068</td>
      <td>971889999</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2197</td>
      <td>개별</td>
      <td>11326510</td>
      <td>혜림감정</td>
      <td>2016-11-22 00:00:00</td>
      <td>3</td>
      <td>2</td>
      <td>149683.1</td>
      <td>71.00</td>
      <td>71.00</td>
      <td>...</td>
      <td>88.0</td>
      <td>NaN</td>
      <td>2017-12-13 00:00:00</td>
      <td>배당</td>
      <td>35.184601</td>
      <td>128.996765</td>
      <td>93399999</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2642</td>
      <td>일반</td>
      <td>183581724</td>
      <td>신라감정</td>
      <td>2016-12-13 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>24405.0</td>
      <td>32.98</td>
      <td>32.98</td>
      <td>...</td>
      <td>110.0</td>
      <td>NaN</td>
      <td>2017-12-27 00:00:00</td>
      <td>배당</td>
      <td>35.154180</td>
      <td>129.089081</td>
      <td>256899000</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1958</td>
      <td>일반</td>
      <td>45887671</td>
      <td>나라감정</td>
      <td>2016-03-07 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>774.0</td>
      <td>45.18</td>
      <td>45.18</td>
      <td>...</td>
      <td>51.0</td>
      <td>NaN</td>
      <td>2016-10-04 00:00:00</td>
      <td>배당</td>
      <td>35.099630</td>
      <td>128.998874</td>
      <td>158660000</td>
      <td>Non-Null</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>



Bid_class _ 입찰 구분 처리


```python
dummy_var = pd.get_dummies(data.Bid_class)
pd.concat([data.drop(['Bid_class'], axis=1),dummy_var], axis=1).head()
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
      <th>Auction_key</th>
      <th>Auction_class</th>
      <th>Claim_price</th>
      <th>Appraisal_company</th>
      <th>Appraisal_date</th>
      <th>Auction_count</th>
      <th>Auction_miscarriage_count</th>
      <th>Total_land_gross_area</th>
      <th>Total_land_real_area</th>
      <th>Total_land_auction_area</th>
      <th>...</th>
      <th>road_bunji2</th>
      <th>Close_date</th>
      <th>Close_result</th>
      <th>point.y</th>
      <th>point.x</th>
      <th>Hammer_price</th>
      <th>road_bunji1_Null</th>
      <th>개별</th>
      <th>일괄</th>
      <th>일반</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2687</td>
      <td>임의</td>
      <td>1766037301</td>
      <td>정명감정</td>
      <td>2017-07-26 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>12592.0</td>
      <td>37.35</td>
      <td>37.35</td>
      <td>...</td>
      <td>NaN</td>
      <td>2018-06-14 00:00:00</td>
      <td>배당</td>
      <td>35.162717</td>
      <td>129.137048</td>
      <td>760000000</td>
      <td>Non-Null</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2577</td>
      <td>임의</td>
      <td>152946867</td>
      <td>희감정</td>
      <td>2016-09-12 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>42478.1</td>
      <td>18.76</td>
      <td>18.76</td>
      <td>...</td>
      <td>NaN</td>
      <td>2017-03-30 00:00:00</td>
      <td>배당</td>
      <td>35.156633</td>
      <td>129.145068</td>
      <td>971889999</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2197</td>
      <td>임의</td>
      <td>11326510</td>
      <td>혜림감정</td>
      <td>2016-11-22 00:00:00</td>
      <td>3</td>
      <td>2</td>
      <td>149683.1</td>
      <td>71.00</td>
      <td>71.00</td>
      <td>...</td>
      <td>NaN</td>
      <td>2017-12-13 00:00:00</td>
      <td>배당</td>
      <td>35.184601</td>
      <td>128.996765</td>
      <td>93399999</td>
      <td>Non-Null</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2642</td>
      <td>임의</td>
      <td>183581724</td>
      <td>신라감정</td>
      <td>2016-12-13 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>24405.0</td>
      <td>32.98</td>
      <td>32.98</td>
      <td>...</td>
      <td>NaN</td>
      <td>2017-12-27 00:00:00</td>
      <td>배당</td>
      <td>35.154180</td>
      <td>129.089081</td>
      <td>256899000</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1958</td>
      <td>강제</td>
      <td>45887671</td>
      <td>나라감정</td>
      <td>2016-03-07 00:00:00</td>
      <td>2</td>
      <td>1</td>
      <td>774.0</td>
      <td>45.18</td>
      <td>45.18</td>
      <td>...</td>
      <td>NaN</td>
      <td>2016-10-04 00:00:00</td>
      <td>배당</td>
      <td>35.099630</td>
      <td>128.998874</td>
      <td>158660000</td>
      <td>Non-Null</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>




```python
data['Bid_class'].unique()
data['Bid_class'].replace(['개별', '일반', '일괄'],[0,1,2]).head(10)
```




    0    0
    1    1
    2    0
    3    1
    4    1
    5    1
    6    1
    7    1
    8    1
    9    1
    Name: Bid_class, dtype: int64



<위치기준>

- 크게 부산, 서울 중 어느 도시에 속하는지 encoding한후,
- 서울에 있는 집은 강북/강남 어디에 속하는지 encoding하려합니다. 


```python
#부산, 서울 column 추가 
dummy_var = pd.get_dummies(data.addr_do)
data=pd.concat([data.drop(['addr_do'], axis=1),dummy_var], axis=1)
```


```python
print(data['addr_si'].unique())
```

    ['해운대구' '사상구' '남구' '사하구' '영도구' '서구' '부산진구' '연제구' '수영구' '동래구' '동구' '중구'
     '강서구' '북구' '금정구' '기장군' '강남구' '은평구' '서초구' '영등포구' '양천구' '마포구' '금천구' '성동구'
     '노원구' '서대문구' '용산구' '구로구' '강북구' '관악구' '송파구' '도봉구' '광진구' '중랑구' '동대문구' '강동구'
     '성북구' '동작구' '종로구']



```python
data['SofHan']=0    #South of Han RIver
data['NofHan']=0    #North of Han River
#강남/강북에 속하는 각 구들을 나열해서 값을 넣습니다.
data.loc[(data.서울==1)&data['addr_si'].isin(['강서구','영등포구','양천구','구로구','동작구','금천구','관악구','서초구','강남구','송파구','강동구']),'SofHan']= 1
data.loc[(data.서울==1)&data['addr_si'].isin(['마포구','서대문구','용산구','중구','종로구','은평구','성북구','동대문구','성동구','광진구','중량구','성북구','강북구','도봉구','노원구']),'NofHan']=1
```


```python
data[(data.서울==1)][['SofHan','NofHan']].groupby('NofHan').count()
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
      <th>SofHan</th>
    </tr>
    <tr>
      <th>NofHan</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>591</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[(data.서울==1)][['SofHan','NofHan']].groupby('SofHan').count().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc7f4110>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_34_1.png)



```python
f, ax = plt.subplots(1, 1, figsize=(7,7))
sns.countplot('서울',hue='SofHan',data=data, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x95f5f50>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_35_1.png)


- 왼쪽이 부산, 오른쪽이 서울입니다.
- 서울에 있는 집은 한강을 기준으로 한번 더 분류해서 표시했습니다. 
- 초록색 막대가 강남에 있는 집입니다. 

## ordinal 변수 4개 

<층 수>
Total_floor
Current_floor

<판매 경력>
Auction_count
Auction_miscarriage_count

## <층수> 변수들 encoding

<층수>
- 층수에 있는 변수들은 속하는 범위에 따라서 다르게 encoding하려고 합니다.

- Total_floor은
- 1~10층 건물 ->'Low_Building'
- 11~20층 건물 -> 'Middle_Building'
- 21~30층 건물 -> 'High_Building'
- 31~40층 건물 -> '2High_Building'
- 41~50층 건물 -> '3High_Building'
- 51~ 층  건물 -> 'SHigh_Building' (Super High)
각 값을 'Height_Building'라는 column을 만들어서, 저장하려합니다. 

- Current_floor은
- 1~10층 집 ->'Low_Floor'
- 11~20층 집 -> 'Middle_Floor'
- 21~30층 집 -> 'High_Floor'
- 31~40층 집 -> '2High_Floor'
- 41~50층 집 -> '3High_Floor'
- 51~ 층  집 -> 'SHigh_Floor' (Super High)
각 값을 'Height_Floor'라는 column을 만들어서, 저장하려합니다. 

- 그리고 float(Current_floor/Total_floor) 값을 이용해서 
각 집이 해당 건물에서, 최대 층 대비 얼마나 높은 층에 위치하는지 encoding하려합니다.
나중에 연속형 변수로 사용하기 위해 'Height_Rate_float'변수를 만들고,

각 값을 'Height_Rate'이라는 column을 만들어서
- 0%~33% -> 'Low_Rate'
- 34%~66% -> 'Middle_Rate'
- 67%~100% -> 'High_Rate'
이렇게 범위를 나눠서 저장하려고 합니다.


```python
data['Height_Building']=''
data['Height_Floor']=''
data['Height_Rate']=''
data['Height_Rate_float']=0
```


```python
data.loc[(data['Total_floor']<11),['Height_Building']]='Low_Building'
data.loc[(data['Total_floor']>10)&(data['Total_floor']<21),['Height_Building']]='Middle_Building'
data.loc[(data['Total_floor']>20)&(data['Total_floor']<31),['Height_Building']]='High_Building'
data.loc[(data['Total_floor']>30)&(data['Total_floor']<41),['Height_Building']]='2High_Building'
data.loc[(data['Total_floor']>40)&(data['Total_floor']<51),['Height_Building']]='3High_Building'
data.loc[(data['Total_floor']>50),['Height_Building']]='SHigh_Building'
```


```python
data['Height_Building'].unique()
data[['Height_Building','Auction_key']].groupby(['Height_Building']).count().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc9a1a90>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_43_1.png)



```python
data.loc[(data['Current_floor']<11),['Height_Floor']]='Low_Floor'
data.loc[(data['Current_floor']>10)&(data['Current_floor']<21),['Height_Floor']]='Middle_Floor'
data.loc[(data['Current_floor']>20)&(data['Current_floor']<31),['Height_Floor']]='High_Floor'
data.loc[(data['Current_floor']>30)&(data['Current_floor']<41),['Height_Floor']]='2High_Floor'
data.loc[(data['Current_floor']>40)&(data['Current_floor']<51),['Height_Floor']]='3High_Floor'
data.loc[(data['Current_floor']>50),['Height_Floor']]='SHigh_Floor'
```


```python
data['Height_Floor'].unique()
```




    array(['Low_Floor', 'SHigh_Floor', 'Middle_Floor', 'High_Floor',
           '2High_Floor', '3High_Floor'], dtype=object)




```python
data[['Height_Floor','Auction_key']].groupby(['Height_Floor']).count().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc8a5850>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_46_1.png)



```python
data['Height_Rate_float']=data['Current_floor']/data['Total_floor']
data.loc[data['Current_floor']/data['Total_floor']<=0.33,['Height_Rate']]='Low_Rate'
data.loc[(data['Current_floor']/data['Total_floor']>0.33)&(data['Current_floor']/data['Total_floor']<=0.66),['Height_Rate']]='Middle_Rate'
data.loc[data['Current_floor']/data['Total_floor']>0.66,['Height_Rate']]='High_Rate'
```


```python
data[['Height_Rate','Auction_key']].groupby(['Height_Rate']).count().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc8f5610>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_48_1.png)


- 생각보다 골고루 분포하네요...!


```python
f, ax = plt.subplots(1,1, figsize=(10,10))
sns.countplot('Height_Building',hue='Height_Rate',data=data,ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc930750>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_50_1.png)


- Building의 높이에 따라, Height_Rate의 분포를 살펴봤습니다!
- 모든 Builiding의 높이에서, High_Rate를 갖는 집들이 가장 많네요!

## <판매 경력> 변수들 encoding

- Auction_count
Auction_miscarriage_count

- 먼저, 이 두 변수 외에도 재매각횟수 변수('Auction_resell_count')를 추가하려합니다. 
(총 판매 횟수) = (유찰 횟수) + (재매각 횟수) + 1 
이 계산식을 이용해서, 재매각 횟수를 구해서 대입하려 합니다. 

- 그리고, 
Auction_Rejected 라는 column을 만들어서 Auction_count의 범위에 따라서 encoding 하려고 합니다.

- Auction_count==1 -> 'Rejected_never'
2<=Auction<=3 -> 'Rejected_few'
4<=Auction<=6 -> 'Rejected_many'
7<=Auction<=13 -> 'Rejected_alot'

### Auction_resell_count 구하기


```python
#(총 판매 횟수) = (유찰 횟수) + (재매각 횟수) + 1 
#1을 더하는 이유는, 현재 경매를 게시하고 있는것을 고려해야하기 때문입니다. 
#재매각 횟수가 어떻게 분포하는지 확인해보겠습니다. 

data['Auction_resell_count']=0
print((data['Auction_count']-data['Auction_miscarriage_count']-1).unique())   #(재매각횟수) = (총 판매 횟수) - (유찰횟수) -1 
data['Auction_resell_count']=data['Auction_count']-data['Auction_miscarriage_count']-1
print(data['Auction_resell_count'].unique())
data[['Auction_count','Auction_resell_count']].groupby(['Auction_resell_count']).count()
```

    [0 1 2 3]
    [0 1 2 3]





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
      <th>Auction_count</th>
    </tr>
    <tr>
      <th>Auction_resell_count</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1847</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Auction_Rejected Encoding


```python
data['Auction_Rejected']=''
data.loc[data['Auction_count']==1,'Auction_Rejected']='Rejected_never'
data.loc[(data['Auction_count']>=2)&(data['Auction_count']<=3),'Auction_Rejected']='Rejected_few'
data.loc[(data['Auction_count']>=4)&(data['Auction_count']<=6),'Auction_Rejected']='Rejected_many'
data.loc[(data['Auction_count']>=7)&(data['Auction_count']<=13),'Auction_Rejected']='Rejected_alot'
```


```python
data[['Auction_Rejected','Auction_count']].groupby(['Auction_Rejected']).count().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0xccb75f0>




![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_58_1.png)


# 과제 3 _ correlation matrix  & Heat Map 생성


```python
numerical_features = list(data.columns)[:-1]
```


```python
data[numerical_features].corr().style.background_gradient('summer_r')
```




<style  type="text/css" >
    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col0 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col1 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col2 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col3 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col4 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col5 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col6 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col7 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col8 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col9 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col10 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col11 {
            background-color:  #c6e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col12 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col13 {
            background-color:  #bbdd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col14 {
            background-color:  #cce666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col15 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col16 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col17 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col18 {
            background-color:  #168b66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col19 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col20 {
            background-color:  #168b66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col21 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col22 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col23 {
            background-color:  #d1e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col24 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col25 {
            background-color:  #fcfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col0 {
            background-color:  #9fcf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col1 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col2 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col3 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col4 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col5 {
            background-color:  #bede66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col6 {
            background-color:  #bfdf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col7 {
            background-color:  #b5da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col8 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col9 {
            background-color:  #91c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col10 {
            background-color:  #86c266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col11 {
            background-color:  #daec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col12 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col13 {
            background-color:  #d7eb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col14 {
            background-color:  #d8ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col15 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col16 {
            background-color:  #bcde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col17 {
            background-color:  #73b966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col18 {
            background-color:  #8dc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col19 {
            background-color:  #84c266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col20 {
            background-color:  #8cc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col21 {
            background-color:  #73b966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col22 {
            background-color:  #81c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col23 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col24 {
            background-color:  #e3f166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col25 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col0 {
            background-color:  #8ec666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col1 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col2 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col3 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col4 {
            background-color:  #fefe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col5 {
            background-color:  #daec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col6 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col7 {
            background-color:  #cbe566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col8 {
            background-color:  #cde666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col9 {
            background-color:  #b3d966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col10 {
            background-color:  #bdde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col11 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col12 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col13 {
            background-color:  #e6f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col14 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col15 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col16 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col17 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col18 {
            background-color:  #79bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col19 {
            background-color:  #b7db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col20 {
            background-color:  #79bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col21 {
            background-color:  #86c266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col22 {
            background-color:  #9dce66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col23 {
            background-color:  #b1d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col24 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col25 {
            background-color:  #6bb566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col0 {
            background-color:  #8dc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col1 {
            background-color:  #ddee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col2 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col3 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col4 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col5 {
            background-color:  #d8ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col6 {
            background-color:  #d9ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col7 {
            background-color:  #c8e366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col8 {
            background-color:  #cae466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col9 {
            background-color:  #b2d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col10 {
            background-color:  #bcde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col11 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col12 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col13 {
            background-color:  #e5f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col14 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col15 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col16 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col17 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col18 {
            background-color:  #79bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col19 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col20 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col21 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col22 {
            background-color:  #9dce66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col23 {
            background-color:  #b1d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col24 {
            background-color:  #eef666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col25 {
            background-color:  #9ece66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col0 {
            background-color:  #8fc766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col1 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col2 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col3 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col4 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col5 {
            background-color:  #ddee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col6 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col7 {
            background-color:  #dfef66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col8 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col9 {
            background-color:  #b9dc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col10 {
            background-color:  #aad466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col11 {
            background-color:  #c9e466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col12 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col13 {
            background-color:  #c5e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col14 {
            background-color:  #cde666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col15 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col16 {
            background-color:  #cbe566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col17 {
            background-color:  #7ebe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col18 {
            background-color:  #82c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col19 {
            background-color:  #a7d366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col20 {
            background-color:  #82c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col21 {
            background-color:  #7dbe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col22 {
            background-color:  #95ca66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col23 {
            background-color:  #aed666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col24 {
            background-color:  #eaf466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col25 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col0 {
            background-color:  #9acc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col1 {
            background-color:  #b8dc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col2 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col3 {
            background-color:  #dcee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col4 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col5 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col6 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col7 {
            background-color:  #0d8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col8 {
            background-color:  #0e8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col9 {
            background-color:  #1e8e66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col10 {
            background-color:  #349a66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col11 {
            background-color:  #d1e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col12 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col13 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col14 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col15 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col16 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col17 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col18 {
            background-color:  #88c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col19 {
            background-color:  #369b66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col20 {
            background-color:  #88c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col21 {
            background-color:  #77bb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col22 {
            background-color:  #8cc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col23 {
            background-color:  #afd766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col24 {
            background-color:  #fcfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col25 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col0 {
            background-color:  #99cc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col1 {
            background-color:  #b8dc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col2 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col3 {
            background-color:  #ddee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col4 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col5 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col6 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col7 {
            background-color:  #0e8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col8 {
            background-color:  #0d8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col9 {
            background-color:  #1d8e66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col10 {
            background-color:  #339966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col11 {
            background-color:  #d0e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col12 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col13 {
            background-color:  #eaf466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col14 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col15 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col16 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col17 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col18 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col19 {
            background-color:  #369b66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col20 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col21 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col22 {
            background-color:  #8dc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col23 {
            background-color:  #afd766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col24 {
            background-color:  #fcfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col25 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col0 {
            background-color:  #9dce66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col1 {
            background-color:  #b3d966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col2 {
            background-color:  #d3e966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col3 {
            background-color:  #d1e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col4 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col5 {
            background-color:  #0d8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col6 {
            background-color:  #0e8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col7 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col8 {
            background-color:  #018066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col9 {
            background-color:  #138966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col10 {
            background-color:  #279366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col11 {
            background-color:  #d6eb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col12 {
            background-color:  #eef666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col13 {
            background-color:  #c3e166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col14 {
            background-color:  #d2e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col15 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col16 {
            background-color:  #c4e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col17 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col18 {
            background-color:  #88c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col19 {
            background-color:  #299466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col20 {
            background-color:  #88c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col21 {
            background-color:  #77bb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col22 {
            background-color:  #8ac466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col23 {
            background-color:  #b1d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col24 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col25 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col0 {
            background-color:  #9cce66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col1 {
            background-color:  #b3d966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col2 {
            background-color:  #d4ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col3 {
            background-color:  #d2e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col4 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col5 {
            background-color:  #0f8766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col6 {
            background-color:  #0d8666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col7 {
            background-color:  #018066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col8 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col9 {
            background-color:  #128866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col10 {
            background-color:  #269266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col11 {
            background-color:  #d5ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col12 {
            background-color:  #eef666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col13 {
            background-color:  #c3e166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col14 {
            background-color:  #d2e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col15 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col16 {
            background-color:  #c6e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col17 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col18 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col19 {
            background-color:  #289366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col20 {
            background-color:  #87c366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col21 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col22 {
            background-color:  #8cc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col23 {
            background-color:  #b0d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col24 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col25 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col0 {
            background-color:  #b9dc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col1 {
            background-color:  #a9d466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col2 {
            background-color:  #dcee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col3 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col4 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col5 {
            background-color:  #249266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col6 {
            background-color:  #249266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col7 {
            background-color:  #168b66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col8 {
            background-color:  #168b66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col9 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col10 {
            background-color:  #078366;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col11 {
            background-color:  #d9ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col12 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col13 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col14 {
            background-color:  #c0e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col15 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col16 {
            background-color:  #bcde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col17 {
            background-color:  #63b166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col18 {
            background-color:  #9cce66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col19 {
            background-color:  #088466;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col20 {
            background-color:  #9dce66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col21 {
            background-color:  #62b066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col22 {
            background-color:  #72b866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col23 {
            background-color:  #afd766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col24 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col25 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col0 {
            background-color:  #c4e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col1 {
            background-color:  #a6d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col2 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col3 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col4 {
            background-color:  #e3f166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col5 {
            background-color:  #43a166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col6 {
            background-color:  #42a066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col7 {
            background-color:  #319866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col8 {
            background-color:  #309866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col9 {
            background-color:  #078366;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col10 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col11 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col12 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col13 {
            background-color:  #aad466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col14 {
            background-color:  #b5da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col15 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col16 {
            background-color:  #bcde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col17 {
            background-color:  #5bad66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col18 {
            background-color:  #a4d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col19 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col20 {
            background-color:  #a5d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col21 {
            background-color:  #5aac66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col22 {
            background-color:  #66b266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col23 {
            background-color:  #b0d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col24 {
            background-color:  #eef666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col25 {
            background-color:  #fbfd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col0 {
            background-color:  #7fbf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col1 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col2 {
            background-color:  #f3f966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col3 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col4 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col5 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col6 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col7 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col8 {
            background-color:  #deee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col9 {
            background-color:  #c0e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col10 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col11 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col12 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col13 {
            background-color:  #c8e366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col14 {
            background-color:  #cfe766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col15 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col16 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col17 {
            background-color:  #8fc766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col18 {
            background-color:  #72b866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col19 {
            background-color:  #b5da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col20 {
            background-color:  #70b866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col21 {
            background-color:  #8fc766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col22 {
            background-color:  #95ca66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col23 {
            background-color:  #c8e366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col24 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col25 {
            background-color:  #f6fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col0 {
            background-color:  #8dc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col1 {
            background-color:  #e2f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col2 {
            background-color:  #eaf466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col3 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col4 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col5 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col6 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col7 {
            background-color:  #e3f166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col8 {
            background-color:  #e4f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col9 {
            background-color:  #c1e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col10 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col11 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col12 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col13 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col14 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col15 {
            background-color:  #fdfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col16 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col17 {
            background-color:  #7ebe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col18 {
            background-color:  #81c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col19 {
            background-color:  #b5da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col20 {
            background-color:  #81c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col21 {
            background-color:  #7ebe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col22 {
            background-color:  #a2d066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col23 {
            background-color:  #a1d066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col24 {
            background-color:  #e5f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col25 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col0 {
            background-color:  #78bc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col1 {
            background-color:  #ddee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col2 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col3 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col4 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col5 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col6 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col7 {
            background-color:  #cbe566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col8 {
            background-color:  #cce666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col9 {
            background-color:  #a0d066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col10 {
            background-color:  #8dc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col11 {
            background-color:  #c8e366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col12 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col13 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col14 {
            background-color:  #42a066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col15 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col16 {
            background-color:  #b3d966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col17 {
            background-color:  #95ca66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col18 {
            background-color:  #6ab466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col19 {
            background-color:  #8cc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col20 {
            background-color:  #6ab466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col21 {
            background-color:  #95ca66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col22 {
            background-color:  #a5d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col23 {
            background-color:  #bdde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col24 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col25 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col0 {
            background-color:  #7ebe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col1 {
            background-color:  #d7eb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col2 {
            background-color:  #f3f966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col3 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col4 {
            background-color:  #dcee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col5 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col6 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col7 {
            background-color:  #d3e966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col8 {
            background-color:  #d5ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col9 {
            background-color:  #a4d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col10 {
            background-color:  #91c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col11 {
            background-color:  #c7e366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col12 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col13 {
            background-color:  #3f9f66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col14 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col15 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col16 {
            background-color:  #cde666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col17 {
            background-color:  #90c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col18 {
            background-color:  #6fb766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col19 {
            background-color:  #90c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col20 {
            background-color:  #6fb766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col21 {
            background-color:  #90c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col22 {
            background-color:  #9fcf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col23 {
            background-color:  #bcde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col24 {
            background-color:  #5aac66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col25 {
            background-color:  #fdfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col0 {
            background-color:  #90c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col1 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col2 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col3 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col4 {
            background-color:  #e6f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col5 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col6 {
            background-color:  #eaf466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col7 {
            background-color:  #e6f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col8 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col9 {
            background-color:  #c0e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col10 {
            background-color:  #b2d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col11 {
            background-color:  #d8ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col12 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col13 {
            background-color:  #d0e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col14 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col15 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col16 {
            background-color:  #d0e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col17 {
            background-color:  #75ba66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col18 {
            background-color:  #89c466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col19 {
            background-color:  #b0d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col20 {
            background-color:  #8ac466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col21 {
            background-color:  #75ba66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col22 {
            background-color:  #9bcd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col23 {
            background-color:  #9fcf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col24 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col25 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col0 {
            background-color:  #90c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col1 {
            background-color:  #b1d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col2 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col3 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col4 {
            background-color:  #cfe766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col5 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col6 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col7 {
            background-color:  #badc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col8 {
            background-color:  #bede66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col9 {
            background-color:  #98cc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col10 {
            background-color:  #8fc766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col11 {
            background-color:  #ddee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col12 {
            background-color:  #dfef66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col13 {
            background-color:  #a4d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col14 {
            background-color:  #c2e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col15 {
            background-color:  #cde666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col16 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col17 {
            background-color:  #7cbe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col18 {
            background-color:  #84c266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col19 {
            background-color:  #8fc766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col20 {
            background-color:  #82c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col21 {
            background-color:  #7dbe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col22 {
            background-color:  #a0d066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col23 {
            background-color:  #aad466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col24 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col25 {
            background-color:  #e4f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col0 {
            background-color:  #fdfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col1 {
            background-color:  #cbe566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col2 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col3 {
            background-color:  #fbfd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col4 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col5 {
            background-color:  #daec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col6 {
            background-color:  #ddee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col7 {
            background-color:  #d5ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col8 {
            background-color:  #d8ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col9 {
            background-color:  #96cb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col10 {
            background-color:  #82c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col11 {
            background-color:  #f6fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col12 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col13 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col14 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col15 {
            background-color:  #d9ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col16 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col17 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col18 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col19 {
            background-color:  #7fbf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col20 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col21 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col22 {
            background-color:  #50a866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col23 {
            background-color:  #52a866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col24 {
            background-color:  #fbfd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col25 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col0 {
            background-color:  #188c66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col1 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col2 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col3 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col4 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col5 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col6 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col7 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col8 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col9 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col10 {
            background-color:  #eaf466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col11 {
            background-color:  #c2e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col12 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col13 {
            background-color:  #b5da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col14 {
            background-color:  #c4e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col15 {
            background-color:  #fefe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col16 {
            background-color:  #f6fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col17 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col18 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col19 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col20 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col21 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col22 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col23 {
            background-color:  #fdfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col24 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col25 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col0 {
            background-color:  #c6e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col1 {
            background-color:  #a4d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col2 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col3 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col4 {
            background-color:  #e2f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col5 {
            background-color:  #46a266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col6 {
            background-color:  #46a266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col7 {
            background-color:  #349a66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col8 {
            background-color:  #349a66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col9 {
            background-color:  #098466;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col10 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col11 {
            background-color:  #dbed66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col12 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col13 {
            background-color:  #aad466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col14 {
            background-color:  #b5da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col15 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col16 {
            background-color:  #bdde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col17 {
            background-color:  #5aac66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col18 {
            background-color:  #a6d266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col19 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col20 {
            background-color:  #a7d366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col21 {
            background-color:  #58ac66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col22 {
            background-color:  #64b266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col23 {
            background-color:  #b0d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col24 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col25 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col0 {
            background-color:  #188c66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col1 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col2 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col3 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col4 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col5 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col6 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col7 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col8 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col9 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col10 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col11 {
            background-color:  #c0e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col12 {
            background-color:  #f1f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col13 {
            background-color:  #b6db66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col14 {
            background-color:  #c6e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col15 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col16 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col17 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col18 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col19 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col20 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col21 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col22 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col23 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col24 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col25 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col0 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col1 {
            background-color:  #cae466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col2 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col3 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col4 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col5 {
            background-color:  #d9ec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col6 {
            background-color:  #dcee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col7 {
            background-color:  #d4ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col8 {
            background-color:  #d7eb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col9 {
            background-color:  #95ca66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col10 {
            background-color:  #80c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col11 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col12 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col13 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col14 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col15 {
            background-color:  #daec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col16 {
            background-color:  #e9f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col17 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col18 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col19 {
            background-color:  #7dbe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col20 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col21 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col22 {
            background-color:  #4ca666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col23 {
            background-color:  #56ab66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col24 {
            background-color:  #fafc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col25 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col0 {
            background-color:  #e5f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col1 {
            background-color:  #badc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col2 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col3 {
            background-color:  #f0f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col4 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col5 {
            background-color:  #d1e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col6 {
            background-color:  #d4ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col7 {
            background-color:  #cae466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col8 {
            background-color:  #cee666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col9 {
            background-color:  #8dc666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col10 {
            background-color:  #77bb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col11 {
            background-color:  #d1e866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col12 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col13 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col14 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col15 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col16 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col17 {
            background-color:  #42a066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col18 {
            background-color:  #c3e166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col19 {
            background-color:  #74ba66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col20 {
            background-color:  #c1e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col21 {
            background-color:  #3e9e66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col22 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col23 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col24 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col25 {
            background-color:  #f3f966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col0 {
            background-color:  #abd566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col1 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col2 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col3 {
            background-color:  #f6fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col4 {
            background-color:  #f8fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col5 {
            background-color:  #eef666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col6 {
            background-color:  #eff766;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col7 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col8 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col9 {
            background-color:  #c6e266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col10 {
            background-color:  #bbdd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col11 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col12 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col13 {
            background-color:  #f2f866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col14 {
            background-color:  #f9fc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col15 {
            background-color:  #dcee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col16 {
            background-color:  #edf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col17 {
            background-color:  #3d9e66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col18 {
            background-color:  #bdde66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col19 {
            background-color:  #b9dc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col20 {
            background-color:  #bfdf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col21 {
            background-color:  #40a066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col22 {
            background-color:  #e2f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col23 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col24 {
            background-color:  #fdfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col25 {
            background-color:  #f6fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col0 {
            background-color:  #89c466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col1 {
            background-color:  #d4ea66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col2 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col3 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col4 {
            background-color:  #ecf666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col5 {
            background-color:  #f3f966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col6 {
            background-color:  #f5fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col7 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col8 {
            background-color:  #eef666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col9 {
            background-color:  #c2e066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col10 {
            background-color:  #b4da66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col11 {
            background-color:  #d6eb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col12 {
            background-color:  #e3f166;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col13 {
            background-color:  #daec66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col14 {
            background-color:  #55aa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col15 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col16 {
            background-color:  #fdfe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col17 {
            background-color:  #85c266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col18 {
            background-color:  #7bbd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col19 {
            background-color:  #b2d866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col20 {
            background-color:  #7abc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col21 {
            background-color:  #85c266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col22 {
            background-color:  #99cc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col23 {
            background-color:  #b3d966;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col24 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col25 {
            background-color:  #f6fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col0 {
            background-color:  #91c866;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col1 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col2 {
            background-color:  #68b366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col3 {
            background-color:  #9acc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col4 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col5 {
            background-color:  #e7f366;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col6 {
            background-color:  #e8f466;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col7 {
            background-color:  #e1f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col8 {
            background-color:  #e2f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col9 {
            background-color:  #bfdf66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col10 {
            background-color:  #bbdd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col11 {
            background-color:  #dcee66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col12 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col13 {
            background-color:  #e5f266;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col14 {
            background-color:  #ebf566;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col15 {
            background-color:  #f7fb66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col16 {
            background-color:  #e0f066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col17 {
            background-color:  #81c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col18 {
            background-color:  #7ebe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col19 {
            background-color:  #b9dc66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col20 {
            background-color:  #7ebe66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col21 {
            background-color:  #81c066;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col22 {
            background-color:  #9bcd66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col23 {
            background-color:  #add666;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col24 {
            background-color:  #f4fa66;
            color:  #000000;
        }    #T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col25 {
            background-color:  #008066;
            color:  #f1f1f1;
        }</style><table id="T_9110e074_b418_11e9_a1cd_982cbcc84294" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Auction_key</th>        <th class="col_heading level0 col1" >Claim_price</th>        <th class="col_heading level0 col2" >Auction_count</th>        <th class="col_heading level0 col3" >Auction_miscarriage_count</th>        <th class="col_heading level0 col4" >Total_land_gross_area</th>        <th class="col_heading level0 col5" >Total_land_real_area</th>        <th class="col_heading level0 col6" >Total_land_auction_area</th>        <th class="col_heading level0 col7" >Total_building_area</th>        <th class="col_heading level0 col8" >Total_building_auction_area</th>        <th class="col_heading level0 col9" >Total_appraisal_price</th>        <th class="col_heading level0 col10" >Minimum_sales_price</th>        <th class="col_heading level0 col11" >addr_bunji1</th>        <th class="col_heading level0 col12" >addr_bunji2</th>        <th class="col_heading level0 col13" >Total_floor</th>        <th class="col_heading level0 col14" >Current_floor</th>        <th class="col_heading level0 col15" >road_bunji1</th>        <th class="col_heading level0 col16" >road_bunji2</th>        <th class="col_heading level0 col17" >point.y</th>        <th class="col_heading level0 col18" >point.x</th>        <th class="col_heading level0 col19" >Hammer_price</th>        <th class="col_heading level0 col20" >부산</th>        <th class="col_heading level0 col21" >서울</th>        <th class="col_heading level0 col22" >SofHan</th>        <th class="col_heading level0 col23" >NofHan</th>        <th class="col_heading level0 col24" >Height_Rate_float</th>        <th class="col_heading level0 col25" >Auction_resell_count</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row0" class="row_heading level0 row0" >Auction_key</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col0" class="data row0 col0" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col1" class="data row0 col1" >-0.136286</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col2" class="data row0 col2" >-0.0205286</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col3" class="data row0 col3" >-0.0130115</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col4" class="data row0 col4" >-0.0212375</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col5" class="data row0 col5" >-0.0994461</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col6" class="data row0 col6" >-0.0934408</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col7" class="data row0 col7" >-0.126003</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col8" class="data row0 col8" >-0.117179</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col9" class="data row0 col9" >-0.322892</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col10" class="data row0 col10" >-0.403807</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col11" class="data row0 col11" >0.0921122</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col12" class="data row0 col12" >-0.013479</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col13" class="data row0 col13" >0.142475</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col14" class="data row0 col14" >0.0977033</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col15" class="data row0 col15" >-0.0318541</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col16" class="data row0 col16" >-0.0346907</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col17" class="data row0 col17" >-0.812046</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col18" class="data row0 col18" >0.828253</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col19" class="data row0 col19" >-0.418769</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col20" class="data row0 col20" >0.827595</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col21" class="data row0 col21" >-0.827595</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col22" class="data row0 col22" >-0.641672</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col23" class="data row0 col23" >-0.220984</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col24" class="data row0 col24" >0.0179438</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row0_col25" class="data row0 col25" >-0.0364459</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row1" class="row_heading level0 row1" >Claim_price</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col0" class="data row1 col0" >-0.136286</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col1" class="data row1 col1" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col2" class="data row1 col2" >0.0133122</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col3" class="data row1 col3" >0.0154113</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col4" class="data row1 col4" >0.00377135</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col5" class="data row1 col5" >0.180421</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col6" class="data row1 col6" >0.182207</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col7" class="data row1 col7" >0.202379</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col8" class="data row1 col8" >0.204</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col9" class="data row1 col9" >0.248846</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col10" class="data row1 col10" >0.261076</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col11" class="data row1 col11" >0.00230263</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col12" class="data row1 col12" >-0.0032264</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col13" class="data row1 col13" >0.0164658</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col14" class="data row1 col14" >0.0432576</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col15" class="data row1 col15" >-0.0380347</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col16" class="data row1 col16" >0.212796</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col17" class="data row1 col17" >0.095939</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col18" class="data row1 col18" >-0.103605</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col19" class="data row1 col19" >0.267728</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col20" class="data row1 col20" >-0.100203</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col21" class="data row1 col21" >0.100203</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col22" class="data row1 col22" >0.172486</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col23" class="data row1 col23" >-0.0639476</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col24" class="data row1 col24" >0.0569834</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row1_col25" class="data row1 col25" >-0.00141048</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row2" class="row_heading level0 row2" >Auction_count</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col0" class="data row2 col0" >-0.0205286</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col1" class="data row2 col1" >0.0133122</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col2" class="data row2 col2" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col3" class="data row2 col3" >0.972918</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col4" class="data row2 col4" >-0.0456972</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col5" class="data row2 col5" >0.0628236</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col6" class="data row2 col6" >0.0628681</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col7" class="data row2 col7" >0.107074</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col8" class="data row2 col8" >0.104286</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col9" class="data row2 col9" >0.0710385</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col10" class="data row2 col10" >-0.0367163</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col11" class="data row2 col11" >-0.0288571</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col12" class="data row2 col12" >0.0106738</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col13" class="data row2 col13" >-0.0523294</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col14" class="data row2 col14" >-0.0291235</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col15" class="data row2 col15" >-0.080393</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col16" class="data row2 col16" >0.00564378</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col17" class="data row2 col17" >-0.0540742</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col18" class="data row2 col18" >0.0516338</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col19" class="data row2 col19" >-0.016999</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col20" class="data row2 col20" >0.0533001</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col21" class="data row2 col21" >-0.0533001</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col22" class="data row2 col22" >-0.00865651</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col23" class="data row2 col23" >-0.0360052</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col24" class="data row2 col24" >0.00595439</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row2_col25" class="data row2 col25" >0.55757</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row3" class="row_heading level0 row3" >Auction_miscarriage_count</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col0" class="data row3 col0" >-0.0130115</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col1" class="data row3 col1" >0.0154113</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col2" class="data row3 col2" >0.972918</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col3" class="data row3 col3" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col4" class="data row3 col4" >-0.0504566</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col5" class="data row3 col5" >0.0695824</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col6" class="data row3 col6" >0.0693916</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col7" class="data row3 col7" >0.11829</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col8" class="data row3 col8" >0.114779</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col9" class="data row3 col9" >0.0775468</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col10" class="data row3 col10" >-0.0330387</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col11" class="data row3 col11" >-0.0302706</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col12" class="data row3 col12" >0.00796616</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col13" class="data row3 col13" >-0.0453428</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col14" class="data row3 col14" >-0.0218523</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col15" class="data row3 col15" >-0.0778034</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col16" class="data row3 col16" >-0.0201408</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col17" class="data row3 col17" >-0.0571973</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col18" class="data row3 col18" >0.0540316</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col19" class="data row3 col19" >-0.011296</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col20" class="data row3 col20" >0.0560915</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col21" class="data row3 col21" >-0.0560915</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col22" class="data row3 col22" >-0.0108045</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col23" class="data row3 col23" >-0.037746</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col24" class="data row3 col24" >0.0101031</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row3_col25" class="data row3 col25" >0.350586</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row4" class="row_heading level0 row4" >Total_land_gross_area</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col0" class="data row4 col0" >-0.0212375</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col1" class="data row4 col1" >0.00377135</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col2" class="data row4 col2" >-0.0456972</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col3" class="data row4 col3" >-0.0504566</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col4" class="data row4 col4" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col5" class="data row4 col5" >0.0497912</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col6" class="data row4 col6" >0.0482252</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col7" class="data row4 col7" >0.0174008</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col8" class="data row4 col8" >0.0164741</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col9" class="data row4 col9" >0.0418959</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col10" class="data row4 col10" >0.0671351</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col11" class="data row4 col11" >0.0805697</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col12" class="data row4 col12" >-0.0201551</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col13" class="data row4 col13" >0.100557</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col14" class="data row4 col14" >0.0943855</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col15" class="data row4 col15" >0.053609</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col16" class="data row4 col16" >0.149867</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col17" class="data row4 col17" >0.0149714</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col18" class="data row4 col18" >-0.0142849</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col19" class="data row4 col19" >0.0704658</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col20" class="data row4 col20" >-0.016506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col21" class="data row4 col21" >0.016506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col22" class="data row4 col22" >0.0443896</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col23" class="data row4 col23" >-0.0196726</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col24" class="data row4 col24" >0.0279761</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row4_col25" class="data row4 col25" >-0.00394257</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row5" class="row_heading level0 row5" >Total_land_real_area</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col0" class="data row5 col0" >-0.0994461</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col1" class="data row5 col1" >0.180421</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col2" class="data row5 col2" >0.0628236</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col3" class="data row5 col3" >0.0695824</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col4" class="data row5 col4" >0.0497912</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col5" class="data row5 col5" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col6" class="data row5 col6" >0.996224</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col7" class="data row5 col7" >0.940361</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col8" class="data row5 col8" >0.934759</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col9" class="data row5 col9" >0.842248</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col10" class="data row5 col10" >0.711515</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col11" class="data row5 col11" >0.0457102</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col12" class="data row5 col12" >0.00525704</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col13" class="data row5 col13" >-0.0725428</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col14" class="data row5 col14" >-0.0626729</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col15" class="data row5 col15" >-0.00329316</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col16" class="data row5 col16" >-0.0694894</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col17" class="data row5 col17" >0.0606016</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col18" class="data row5 col18" >-0.0626211</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col19" class="data row5 col19" >0.696099</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col20" class="data row5 col20" >-0.0640388</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col21" class="data row5 col21" >0.0640388</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col22" class="data row5 col22" >0.0993394</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col23" class="data row5 col23" >-0.0259049</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col24" class="data row5 col24" >-0.0462272</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row5_col25" class="data row5 col25" >0.00464564</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row6" class="row_heading level0 row6" >Total_land_auction_area</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col0" class="data row6 col0" >-0.0934408</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col1" class="data row6 col1" >0.182207</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col2" class="data row6 col2" >0.0628681</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col3" class="data row6 col3" >0.0693916</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col4" class="data row6 col4" >0.0482252</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col5" class="data row6 col5" >0.996224</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col6" class="data row6 col6" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col7" class="data row6 col7" >0.938144</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col8" class="data row6 col8" >0.941681</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col9" class="data row6 col9" >0.845243</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col10" class="data row6 col10" >0.715891</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col11" class="data row6 col11" >0.0500148</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col12" class="data row6 col12" >0.00498476</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col13" class="data row6 col13" >-0.0683851</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col14" class="data row6 col14" >-0.0612739</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col15" class="data row6 col15" >-0.00221032</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col16" class="data row6 col16" >-0.069589</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col17" class="data row6 col17" >0.0557761</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col18" class="data row6 col18" >-0.0577958</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col19" class="data row6 col19" >0.700639</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col20" class="data row6 col20" >-0.0590506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col21" class="data row6 col21" >0.0590506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col22" class="data row6 col22" >0.0919366</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col23" class="data row6 col23" >-0.023919</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col24" class="data row6 col24" >-0.0479423</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row6_col25" class="data row6 col25" >0.00551099</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row7" class="row_heading level0 row7" >Total_building_area</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col0" class="data row7 col0" >-0.126003</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col1" class="data row7 col1" >0.202379</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col2" class="data row7 col2" >0.107074</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col3" class="data row7 col3" >0.11829</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col4" class="data row7 col4" >0.0174008</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col5" class="data row7 col5" >0.940361</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col6" class="data row7 col6" >0.938144</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col7" class="data row7 col7" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col8" class="data row7 col8" >0.993533</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col9" class="data row7 col9" >0.900302</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col10" class="data row7 col10" >0.783299</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col11" class="data row7 col11" >0.0224659</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col12" class="data row7 col12" >0.00153488</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col13" class="data row7 col13" >0.107069</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col14" class="data row7 col14" >0.0711099</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col15" class="data row7 col15" >-0.0119606</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col16" class="data row7 col16" >0.180631</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col17" class="data row7 col17" >0.0627975</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col18" class="data row7 col18" >-0.0645719</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col19" class="data row7 col19" >0.768454</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col20" class="data row7 col20" >-0.0674114</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col21" class="data row7 col21" >0.0674114</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col22" class="data row7 col22" >0.109469</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col23" class="data row7 col23" >-0.0337912</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col24" class="data row7 col24" >-0.0362445</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row7_col25" class="data row7 col25" >0.0090044</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row8" class="row_heading level0 row8" >Total_building_auction_area</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col0" class="data row8 col0" >-0.117179</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col1" class="data row8 col1" >0.204</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col2" class="data row8 col2" >0.104286</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col3" class="data row8 col3" >0.114779</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col4" class="data row8 col4" >0.0164741</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col5" class="data row8 col5" >0.934759</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col6" class="data row8 col6" >0.941681</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col7" class="data row8 col7" >0.993533</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col8" class="data row8 col8" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col9" class="data row8 col9" >0.901988</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col10" class="data row8 col10" >0.787366</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col11" class="data row8 col11" >0.0274295</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col12" class="data row8 col12" >0.00198505</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col13" class="data row8 col13" >0.108707</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col14" class="data row8 col14" >0.0698194</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col15" class="data row8 col15" >-0.00904795</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col16" class="data row8 col16" >0.170577</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col17" class="data row8 col17" >0.0557297</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col18" class="data row8 col18" >-0.0575015</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col19" class="data row8 col19" >0.772839</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col20" class="data row8 col20" >-0.0601176</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col21" class="data row8 col21" >0.0601176</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col22" class="data row8 col22" >0.100581</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col23" class="data row8 col23" >-0.0329698</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col24" class="data row8 col24" >-0.038731</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row8_col25" class="data row8 col25" >0.0103222</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row9" class="row_heading level0 row9" >Total_appraisal_price</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col0" class="data row9 col0" >-0.322892</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col1" class="data row9 col1" >0.248846</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col2" class="data row9 col2" >0.0710385</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col3" class="data row9 col3" >0.0775468</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col4" class="data row9 col4" >0.0418959</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col5" class="data row9 col5" >0.842248</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col6" class="data row9 col6" >0.845243</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col7" class="data row9 col7" >0.900302</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col8" class="data row9 col8" >0.901988</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col9" class="data row9 col9" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col10" class="data row9 col10" >0.960357</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col11" class="data row9 col11" >0.00572622</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col12" class="data row9 col12" >-0.00135566</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col13" class="data row9 col13" >0.169577</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col14" class="data row9 col14" >0.151757</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col15" class="data row9 col15" >0.00745064</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col16" class="data row9 col16" >0.214252</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col17" class="data row9 col17" >0.220942</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col18" class="data row9 col18" >-0.22211</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col19" class="data row9 col19" >0.953464</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col20" class="data row9 col20" >-0.229401</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col21" class="data row9 col21" >0.229401</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col22" class="data row9 col22" >0.268838</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col23" class="data row9 col23" >-0.0245696</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col24" class="data row9 col24" >-0.0025341</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row9_col25" class="data row9 col25" >0.00932649</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row10" class="row_heading level0 row10" >Minimum_sales_price</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col0" class="data row10 col0" >-0.403807</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col1" class="data row10 col1" >0.261076</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col2" class="data row10 col2" >-0.0367163</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col3" class="data row10 col3" >-0.0330387</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col4" class="data row10 col4" >0.0671351</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col5" class="data row10 col5" >0.711515</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col6" class="data row10 col6" >0.715891</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col7" class="data row10 col7" >0.783299</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col8" class="data row10 col8" >0.787366</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col9" class="data row10 col9" >0.960357</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col10" class="data row10 col10" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col11" class="data row10 col11" >-0.000993686</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col12" class="data row10 col12" >-0.00278806</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col13" class="data row10 col13" >0.223407</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col14" class="data row10 col14" >0.201913</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col15" class="data row10 col15" >0.0217761</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col16" class="data row10 col16" >0.211297</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col17" class="data row10 col17" >0.285141</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col18" class="data row10 col18" >-0.285804</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col19" class="data row10 col19" >0.994592</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col20" class="data row10 col20" >-0.295578</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col21" class="data row10 col21" >0.295578</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col22" class="data row10 col22" >0.34397</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col23" class="data row10 col23" >-0.0302089</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col24" class="data row10 col24" >0.0110991</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row10_col25" class="data row10 col25" >-0.0301083</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row11" class="row_heading level0 row11" >addr_bunji1</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col0" class="data row11 col0" >0.0921122</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col1" class="data row11 col1" >0.00230263</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col2" class="data row11 col2" >-0.0288571</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col3" class="data row11 col3" >-0.0302706</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col4" class="data row11 col4" >0.0805697</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col5" class="data row11 col5" >0.0457102</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col6" class="data row11 col6" >0.0500148</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col7" class="data row11 col7" >0.0224659</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col8" class="data row11 col8" >0.0274295</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col9" class="data row11 col9" >0.00572622</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col10" class="data row11 col10" >-0.000993686</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col11" class="data row11 col11" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col12" class="data row11 col12" >-0.0251076</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col13" class="data row11 col13" >0.0856667</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col14" class="data row11 col14" >0.0870213</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col15" class="data row11 col15" >0.0132009</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col16" class="data row11 col16" >-0.0106838</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col17" class="data row11 col17" >-0.12324</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col18" class="data row11 col18" >0.109759</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col19" class="data row11 col19" >-0.0031392</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col20" class="data row11 col20" >0.119691</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col21" class="data row11 col21" >-0.119691</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col22" class="data row11 col22" >0.0426612</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col23" class="data row11 col23" >-0.168856</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col24" class="data row11 col24" >0.0184908</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row11_col25" class="data row11 col25" >-0.00821118</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row12" class="row_heading level0 row12" >addr_bunji2</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col0" class="data row12 col0" >-0.013479</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col1" class="data row12 col1" >-0.0032264</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col2" class="data row12 col2" >0.0106738</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col3" class="data row12 col3" >0.00796616</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col4" class="data row12 col4" >-0.0201551</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col5" class="data row12 col5" >0.00525704</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col6" class="data row12 col6" >0.00498476</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col7" class="data row12 col7" >0.00153488</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col8" class="data row12 col8" >0.00198505</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col9" class="data row12 col9" >-0.00135566</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col10" class="data row12 col10" >-0.00278806</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col11" class="data row12 col11" >-0.0251076</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col12" class="data row12 col12" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col13" class="data row12 col13" >-0.0029867</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col14" class="data row12 col14" >0.0308223</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col15" class="data row12 col15" >-0.0703143</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col16" class="data row12 col16" >0.0637342</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col17" class="data row12 col17" >0.00877516</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col18" class="data row12 col18" >-0.0100133</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col19" class="data row12 col19" >-0.00364121</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col20" class="data row12 col20" >-0.00912144</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col21" class="data row12 col21" >0.00912144</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col22" class="data row12 col22" >-0.0403837</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col23" class="data row12 col23" >0.0590072</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col24" class="data row12 col24" >0.0483241</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row12_col25" class="data row12 col25" >0.0135081</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row13" class="row_heading level0 row13" >Total_floor</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col0" class="data row13 col0" >0.142475</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col1" class="data row13 col1" >0.0164658</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col2" class="data row13 col2" >-0.0523294</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col3" class="data row13 col3" >-0.0453428</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col4" class="data row13 col4" >0.100557</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col5" class="data row13 col5" >-0.0725428</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col6" class="data row13 col6" >-0.0683851</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col7" class="data row13 col7" >0.107069</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col8" class="data row13 col8" >0.108707</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col9" class="data row13 col9" >0.169577</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col10" class="data row13 col10" >0.223407</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col11" class="data row13 col11" >0.0856667</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col12" class="data row13 col12" >-0.0029867</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col13" class="data row13 col13" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col14" class="data row13 col14" >0.708154</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col15" class="data row13 col15" >0.0486137</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col16" class="data row13 col16" >0.248265</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col17" class="data row13 col17" >-0.167823</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col18" class="data row13 col18" >0.172578</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col19" class="data row13 col19" >0.22083</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col20" class="data row13 col20" >0.165918</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col21" class="data row13 col21" >-0.165918</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col22" class="data row13 col22" >-0.0592911</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col23" class="data row13 col23" >-0.107802</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col24" class="data row13 col24" >0.00186703</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row13_col25" class="data row13 col25" >-0.0491789</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row14" class="row_heading level0 row14" >Current_floor</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col0" class="data row14 col0" >0.0977033</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col1" class="data row14 col1" >0.0432576</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col2" class="data row14 col2" >-0.0291235</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col3" class="data row14 col3" >-0.0218523</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col4" class="data row14 col4" >0.0943855</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col5" class="data row14 col5" >-0.0626729</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col6" class="data row14 col6" >-0.0612739</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col7" class="data row14 col7" >0.0711099</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col8" class="data row14 col8" >0.0698194</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col9" class="data row14 col9" >0.151757</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col10" class="data row14 col10" >0.201913</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col11" class="data row14 col11" >0.0870213</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col12" class="data row14 col12" >0.0308223</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col13" class="data row14 col13" >0.708154</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col14" class="data row14 col14" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col15" class="data row14 col15" >0.00594338</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col16" class="data row14 col16" >0.142646</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col17" class="data row14 col17" >-0.128727</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col18" class="data row14 col18" >0.131782</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col19" class="data row14 col19" >0.198013</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col20" class="data row14 col20" >0.126812</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col21" class="data row14 col21" >-0.126812</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col22" class="data row14 col22" >-0.0233762</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col23" class="data row14 col23" >-0.0998784</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col24" class="data row14 col24" >0.625042</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row14_col25" class="data row14 col25" >-0.0395191</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row15" class="row_heading level0 row15" >road_bunji1</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col0" class="data row15 col0" >-0.0318541</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col1" class="data row15 col1" >-0.0380347</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col2" class="data row15 col2" >-0.080393</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col3" class="data row15 col3" >-0.0778034</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col4" class="data row15 col4" >0.053609</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col5" class="data row15 col5" >-0.00329316</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col6" class="data row15 col6" >-0.00221032</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col7" class="data row15 col7" >-0.0119606</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col8" class="data row15 col8" >-0.00904795</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col9" class="data row15 col9" >0.00745064</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col10" class="data row15 col10" >0.0217761</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col11" class="data row15 col11" >0.0132009</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col12" class="data row15 col12" >-0.0703143</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col13" class="data row15 col13" >0.0486137</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col14" class="data row15 col14" >0.00594338</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col15" class="data row15 col15" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col16" class="data row15 col16" >0.130891</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col17" class="data row15 col17" >0.0811103</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col18" class="data row15 col18" >-0.0748654</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col19" class="data row15 col19" >0.0215739</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col20" class="data row15 col20" >-0.0790701</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col21" class="data row15 col21" >0.0790701</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col22" class="data row15 col22" >0.000631759</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col23" class="data row15 col23" >0.0684746</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col24" class="data row15 col24" >-0.0314623</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row15_col25" class="data row15 col25" >-0.0463061</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row16" class="row_heading level0 row16" >road_bunji2</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col0" class="data row16 col0" >-0.0346907</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col1" class="data row16 col1" >0.212796</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col2" class="data row16 col2" >0.00564378</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col3" class="data row16 col3" >-0.0201408</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col4" class="data row16 col4" >0.149867</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col5" class="data row16 col5" >-0.0694894</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col6" class="data row16 col6" >-0.069589</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col7" class="data row16 col7" >0.180631</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col8" class="data row16 col8" >0.170577</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col9" class="data row16 col9" >0.214252</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col10" class="data row16 col10" >0.211297</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col11" class="data row16 col11" >-0.0106838</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col12" class="data row16 col12" >0.0637342</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col13" class="data row16 col13" >0.248265</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col14" class="data row16 col14" >0.142646</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col15" class="data row16 col15" >0.130891</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col16" class="data row16 col16" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col17" class="data row16 col17" >0.027675</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col18" class="data row16 col18" >-0.0312271</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col19" class="data row16 col19" >0.206681</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col20" class="data row16 col20" >-0.0233703</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col21" class="data row16 col21" >0.0233703</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col22" class="data row16 col22" >-0.0263594</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col23" class="data row16 col23" >0.00566105</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col24" class="data row16 col24" >-0.0605171</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row16_col25" class="data row16 col25" >0.0636646</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row17" class="row_heading level0 row17" >point.y</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col0" class="data row17 col0" >-0.812046</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col1" class="data row17 col1" >0.095939</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col2" class="data row17 col2" >-0.0540742</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col3" class="data row17 col3" >-0.0571973</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col4" class="data row17 col4" >0.0149714</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col5" class="data row17 col5" >0.0606016</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col6" class="data row17 col6" >0.0557761</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col7" class="data row17 col7" >0.0627975</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col8" class="data row17 col8" >0.0557297</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col9" class="data row17 col9" >0.220942</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col10" class="data row17 col10" >0.285141</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col11" class="data row17 col11" >-0.12324</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col12" class="data row17 col12" >0.00877516</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col13" class="data row17 col13" >-0.167823</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col14" class="data row17 col14" >-0.128727</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col15" class="data row17 col15" >0.0811103</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col16" class="data row17 col16" >0.027675</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col17" class="data row17 col17" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col18" class="data row17 col18" >-0.994182</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col19" class="data row17 col19" >0.295698</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col20" class="data row17 col20" >-0.998749</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col21" class="data row17 col21" >0.998749</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col22" class="data row17 col22" >0.480567</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col23" class="data row17 col23" >0.520766</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col24" class="data row17 col24" >-0.0413016</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row17_col25" class="data row17 col25" >-0.0136749</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row18" class="row_heading level0 row18" >point.x</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col0" class="data row18 col0" >0.828253</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col1" class="data row18 col1" >-0.103605</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col2" class="data row18 col2" >0.0516338</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col3" class="data row18 col3" >0.0540316</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col4" class="data row18 col4" >-0.0142849</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col5" class="data row18 col5" >-0.0626211</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col6" class="data row18 col6" >-0.0577958</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col7" class="data row18 col7" >-0.0645719</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col8" class="data row18 col8" >-0.0575015</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col9" class="data row18 col9" >-0.22211</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col10" class="data row18 col10" >-0.285804</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col11" class="data row18 col11" >0.109759</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col12" class="data row18 col12" >-0.0100133</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col13" class="data row18 col13" >0.172578</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col14" class="data row18 col14" >0.131782</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col15" class="data row18 col15" >-0.0748654</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col16" class="data row18 col16" >-0.0312271</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col17" class="data row18 col17" >-0.994182</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col18" class="data row18 col18" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col19" class="data row18 col19" >-0.296312</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col20" class="data row18 col20" >0.99685</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col21" class="data row18 col21" >-0.99685</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col22" class="data row18 col22" >-0.523997</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col23" class="data row18 col23" >-0.480567</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col24" class="data row18 col24" >0.0391176</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row18_col25" class="data row18 col25" >0.0151562</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row19" class="row_heading level0 row19" >Hammer_price</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col0" class="data row19 col0" >-0.418769</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col1" class="data row19 col1" >0.267728</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col2" class="data row19 col2" >-0.016999</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col3" class="data row19 col3" >-0.011296</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col4" class="data row19 col4" >0.0704658</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col5" class="data row19 col5" >0.696099</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col6" class="data row19 col6" >0.700639</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col7" class="data row19 col7" >0.768454</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col8" class="data row19 col8" >0.772839</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col9" class="data row19 col9" >0.953464</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col10" class="data row19 col10" >0.994592</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col11" class="data row19 col11" >-0.0031392</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col12" class="data row19 col12" >-0.00364121</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col13" class="data row19 col13" >0.22083</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col14" class="data row19 col14" >0.198013</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col15" class="data row19 col15" >0.0215739</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col16" class="data row19 col16" >0.206681</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col17" class="data row19 col17" >0.295698</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col18" class="data row19 col18" >-0.296312</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col19" class="data row19 col19" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col20" class="data row19 col20" >-0.306488</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col21" class="data row19 col21" >0.306488</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col22" class="data row19 col22" >0.355196</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col23" class="data row19 col23" >-0.0292271</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col24" class="data row19 col24" >0.00858097</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row19_col25" class="data row19 col25" >-0.0283061</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row20" class="row_heading level0 row20" >부산</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col0" class="data row20 col0" >0.827595</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col1" class="data row20 col1" >-0.100203</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col2" class="data row20 col2" >0.0533001</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col3" class="data row20 col3" >0.0560915</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col4" class="data row20 col4" >-0.016506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col5" class="data row20 col5" >-0.0640388</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col6" class="data row20 col6" >-0.0590506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col7" class="data row20 col7" >-0.0674114</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col8" class="data row20 col8" >-0.0601176</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col9" class="data row20 col9" >-0.229401</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col10" class="data row20 col10" >-0.295578</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col11" class="data row20 col11" >0.119691</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col12" class="data row20 col12" >-0.00912144</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col13" class="data row20 col13" >0.165918</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col14" class="data row20 col14" >0.126812</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col15" class="data row20 col15" >-0.0790701</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col16" class="data row20 col16" >-0.0233703</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col17" class="data row20 col17" >-0.998749</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col18" class="data row20 col18" >0.99685</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col19" class="data row20 col19" >-0.306488</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col20" class="data row20 col20" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col21" class="data row20 col21" >-1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col22" class="data row20 col22" >-0.508909</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col23" class="data row20 col23" >-0.494989</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col24" class="data row20 col24" >0.0396385</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row20_col25" class="data row20 col25" >0.01451</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row21" class="row_heading level0 row21" >서울</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col0" class="data row21 col0" >-0.827595</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col1" class="data row21 col1" >0.100203</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col2" class="data row21 col2" >-0.0533001</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col3" class="data row21 col3" >-0.0560915</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col4" class="data row21 col4" >0.016506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col5" class="data row21 col5" >0.0640388</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col6" class="data row21 col6" >0.0590506</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col7" class="data row21 col7" >0.0674114</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col8" class="data row21 col8" >0.0601176</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col9" class="data row21 col9" >0.229401</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col10" class="data row21 col10" >0.295578</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col11" class="data row21 col11" >-0.119691</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col12" class="data row21 col12" >0.00912144</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col13" class="data row21 col13" >-0.165918</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col14" class="data row21 col14" >-0.126812</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col15" class="data row21 col15" >0.0790701</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col16" class="data row21 col16" >0.0233703</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col17" class="data row21 col17" >0.998749</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col18" class="data row21 col18" >-0.99685</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col19" class="data row21 col19" >0.306488</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col20" class="data row21 col20" >-1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col21" class="data row21 col21" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col22" class="data row21 col22" >0.508909</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col23" class="data row21 col23" >0.494989</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col24" class="data row21 col24" >-0.0396385</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row21_col25" class="data row21 col25" >-0.01451</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row22" class="row_heading level0 row22" >SofHan</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col0" class="data row22 col0" >-0.641672</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col1" class="data row22 col1" >0.172486</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col2" class="data row22 col2" >-0.00865651</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col3" class="data row22 col3" >-0.0108045</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col4" class="data row22 col4" >0.0443896</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col5" class="data row22 col5" >0.0993394</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col6" class="data row22 col6" >0.0919366</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col7" class="data row22 col7" >0.109469</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col8" class="data row22 col8" >0.100581</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col9" class="data row22 col9" >0.268838</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col10" class="data row22 col10" >0.34397</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col11" class="data row22 col11" >0.0426612</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col12" class="data row22 col12" >-0.0403837</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col13" class="data row22 col13" >-0.0592911</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col14" class="data row22 col14" >-0.0233762</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col15" class="data row22 col15" >0.000631759</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col16" class="data row22 col16" >-0.0263594</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col17" class="data row22 col17" >0.480567</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col18" class="data row22 col18" >-0.523997</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col19" class="data row22 col19" >0.355196</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col20" class="data row22 col20" >-0.508909</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col21" class="data row22 col21" >0.508909</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col22" class="data row22 col22" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col23" class="data row22 col23" >-0.452772</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col24" class="data row22 col24" >0.0150136</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row22_col25" class="data row22 col25" >0.00372926</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row23" class="row_heading level0 row23" >NofHan</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col0" class="data row23 col0" >-0.220984</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col1" class="data row23 col1" >-0.0639476</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col2" class="data row23 col2" >-0.0360052</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col3" class="data row23 col3" >-0.037746</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col4" class="data row23 col4" >-0.0196726</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col5" class="data row23 col5" >-0.0259049</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col6" class="data row23 col6" >-0.023919</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col7" class="data row23 col7" >-0.0337912</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col8" class="data row23 col8" >-0.0329698</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col9" class="data row23 col9" >-0.0245696</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col10" class="data row23 col10" >-0.0302089</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col11" class="data row23 col11" >-0.168856</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col12" class="data row23 col12" >0.0590072</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col13" class="data row23 col13" >-0.107802</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col14" class="data row23 col14" >-0.0998784</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col15" class="data row23 col15" >0.0684746</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col16" class="data row23 col16" >0.00566105</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col17" class="data row23 col17" >0.520766</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col18" class="data row23 col18" >-0.480567</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col19" class="data row23 col19" >-0.0292271</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col20" class="data row23 col20" >-0.494989</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col21" class="data row23 col21" >0.494989</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col22" class="data row23 col22" >-0.452772</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col23" class="data row23 col23" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col24" class="data row23 col24" >-0.0488623</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row23_col25" class="data row23 col25" >-0.0103218</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row24" class="row_heading level0 row24" >Height_Rate_float</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col0" class="data row24 col0" >0.0179438</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col1" class="data row24 col1" >0.0569834</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col2" class="data row24 col2" >0.00595439</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col3" class="data row24 col3" >0.0101031</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col4" class="data row24 col4" >0.0279761</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col5" class="data row24 col5" >-0.0462272</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col6" class="data row24 col6" >-0.0479423</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col7" class="data row24 col7" >-0.0362445</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col8" class="data row24 col8" >-0.038731</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col9" class="data row24 col9" >-0.0025341</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col10" class="data row24 col10" >0.0110991</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col11" class="data row24 col11" >0.0184908</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col12" class="data row24 col12" >0.0483241</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col13" class="data row24 col13" >0.00186703</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col14" class="data row24 col14" >0.625042</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col15" class="data row24 col15" >-0.0314623</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col16" class="data row24 col16" >-0.0605171</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col17" class="data row24 col17" >-0.0413016</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col18" class="data row24 col18" >0.0391176</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col19" class="data row24 col19" >0.00858097</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col20" class="data row24 col20" >0.0396385</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col21" class="data row24 col21" >-0.0396385</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col22" class="data row24 col22" >0.0150136</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col23" class="data row24 col23" >-0.0488623</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col24" class="data row24 col24" >1</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row24_col25" class="data row24 col25" >-0.0121585</td>
            </tr>
            <tr>
                        <th id="T_9110e074_b418_11e9_a1cd_982cbcc84294level0_row25" class="row_heading level0 row25" >Auction_resell_count</th>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col0" class="data row25 col0" >-0.0364459</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col1" class="data row25 col1" >-0.00141048</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col2" class="data row25 col2" >0.55757</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col3" class="data row25 col3" >0.350586</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col4" class="data row25 col4" >-0.00394257</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col5" class="data row25 col5" >0.00464564</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col6" class="data row25 col6" >0.00551099</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col7" class="data row25 col7" >0.0090044</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col8" class="data row25 col8" >0.0103222</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col9" class="data row25 col9" >0.00932649</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col10" class="data row25 col10" >-0.0301083</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col11" class="data row25 col11" >-0.00821118</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col12" class="data row25 col12" >0.0135081</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col13" class="data row25 col13" >-0.0491789</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col14" class="data row25 col14" >-0.0395191</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col15" class="data row25 col15" >-0.0463061</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col16" class="data row25 col16" >0.0636646</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col17" class="data row25 col17" >-0.0136749</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col18" class="data row25 col18" >0.0151562</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col19" class="data row25 col19" >-0.0283061</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col20" class="data row25 col20" >0.01451</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col21" class="data row25 col21" >-0.01451</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col22" class="data row25 col22" >0.00372926</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col23" class="data row25 col23" >-0.0103218</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col24" class="data row25 col24" >-0.0121585</td>
                        <td id="T_9110e074_b418_11e9_a1cd_982cbcc84294row25_col25" class="data row25 col25" >1</td>
            </tr>
    </tbody></table>




```python
#보기 힘들어서 csv파일로 저장 해서 확인해봤습니다.
data[numerical_features].corr().to_csv("correlation.csv", mode='w')
```

연속변수를 포함해서 다양한 변수들의 correlation을 확인해봤습니다

상관관계가 큰 변수들은 

- <경매 횟수 변수들>
Auction_count , Auction_miscarriage_count
 
 유찰횟수가 클 수록 경매횟수가 클 수밖에 없기 때문에, 두 변수간에는 큰 상관관계가 존재합니다. 

- <면적 관련 변수들>
Total_land_real_area 
Total_land_auction_area
Total_building_area
Total_building_auction_area
- <집 가격 변수들>
Total_appraisal_price
Minimum_sales_price	
-<지역 변수들>
부산
서울
강남
강북 

집이 클 수록, 그 집을 포함한 건물도 클 수 밖에 없으므로, 면적과 관련된 변수들끼리는 큰 상관관계가 존재합니다. 
그리고 통상적으로, 집 면적이 커질 수록, 가격 또한 오르기 때문에, 큰 상관관계가 존재합니다. 


                   부산     서울        강남        강북
    Hammer_ Price -0.306488	0.306488	0.355196	-0.0292271

부산이 서울과 비교했을때 확실히, 음의 상관관계를 가지고, 
강남 지역 집들이 강북 지역 집들보다 가격이 비쌌다고 예상할 수 있습니다. 
0과 1의 값으로 encoding 했기 떄문에, 상관계수 값이 뚜렷하지 않은것 같습니다. 


```python
numerical_features
```




    ['Auction_key',
     'Auction_class',
     'Bid_class',
     'Claim_price',
     'Appraisal_company',
     'Appraisal_date',
     'Auction_count',
     'Auction_miscarriage_count',
     'Total_land_gross_area',
     'Total_land_real_area',
     'Total_land_auction_area',
     'Total_building_area',
     'Total_building_auction_area',
     'Total_appraisal_price',
     'Minimum_sales_price',
     'First_auction_date',
     'Final_auction_date',
     'Final_result',
     'Creditor',
     'addr_si',
     'addr_dong',
     'addr_li',
     'addr_san',
     'addr_bunji1',
     'addr_bunji2',
     'addr_etc',
     'Apartment_usage',
     'Preserve_regist_date',
     'Total_floor',
     'Current_floor',
     'Specific',
     'Share_auction_YorN',
     'road_name',
     'road_bunji1',
     'road_bunji2',
     'Close_date',
     'Close_result',
     'point.y',
     'point.x',
     'Hammer_price',
     'road_bunji1_Null',
     '부산',
     '서울',
     'SofHan',
     'NofHan',
     'Height_Building',
     'Height_Floor',
     'Height_Rate',
     'Height_Rate_float',
     'Auction_resell_count']




```python
sns.heatmap(data[numerical_features].corr(),annot=True,linewidths=0.1) 
fig=plt.gcf()
fig.set_size_inches(30,30)
plt.show()
```


![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_65_0.png)


# 연속형 변수 5개 선정 & 타겟 변수와의 관계 표현 

- Claim_price
- Minimum_sales_price
- Total_land_auction_area
- Total_building_area
- point_y 
- point_x
- Height_Rate_float

## Claim_price와 Hammer_price의 관계 표현


```python
data[(data['Claim_price']<1000000)][['Claim_price','Hammer_price']].describe()
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
      <th>Claim_price</th>
      <th>Hammer_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22.000000</td>
      <td>2.200000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14819.227273</td>
      <td>6.801045e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>48601.680977</td>
      <td>5.595434e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.921000e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>3.141050e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>5.253439e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>8.853250e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>200000.000000</td>
      <td>2.086600e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[['Claim_price','Hammer_price']].describe()
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
      <th>Claim_price</th>
      <th>Hammer_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.933000e+03</td>
      <td>1.933000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.703908e+08</td>
      <td>4.726901e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.337869e+09</td>
      <td>5.574493e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>6.303000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.746112e+07</td>
      <td>1.975550e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.728143e+08</td>
      <td>3.544500e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.565089e+08</td>
      <td>5.599000e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.286481e+10</td>
      <td>1.515100e+10</td>
    </tr>
  </tbody>
</table>
</div>



- Claim_price의 최소값이 0이고, 천만원보다 적은 수를 갖는 데이터가 많은 것을 보면 
- dummy값이 많이 있는것을 알 수 있습니다.
- 그리고, 평균이 3억 7천이므로
- 1억 이하의 집은 제외하고 관계를 표현하겠습니다. 


```python
f,ax=plt.subplots(1,1,figsize=(15,15))
# sns.relplot(x="Claim_price", y="Hammer_price", data=data,ax=ax)
plt.scatter(data['Claim_price'], data['Hammer_price'], color = 'gray', s=10) # s controls point size
plt.xlim(100000000,600000000)
plt.ylim(100000000,600000000)
x = np.linspace(0, 600000000, 6)
plt.plot(x, x, label='linear')
ax.set_ylim(100000000,600000000)
ax.set_yticks(range(100000000,600000000,25000000))
ax.set_xlim(100000000,600000000)
ax.set_xticks(range(100000000,600000000,25000000))
plt.show()
```


![png](Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_files/Week1_EDA_HW2_%EA%B8%B8%ED%83%9C%ED%98%95_72_0.png)


- Claim_price와 Hammer_price를 비교하기 위해, y=x 그래프를 그렸습니다. 
- y=x 선보다 위에 존재하는 점들이 월등히 많습니다. 즉, 많은 집들이 초기 예상 가격보다 많은 값을 받은것입니다.


```python
data[['Claim_price','Hammer_price']]
```


```python
data = [5,8,12,18,19,19.9,20.1,21,24,28] 

fig, ax = plt.subplots()
sns.distplot(data, ax=ax)
ax.set_xlim(0,31)
ax.set_xticks(range(1,32,3))
plt.show()
```


```python

```
