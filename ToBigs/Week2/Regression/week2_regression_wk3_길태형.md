```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

random.seed(1)
```


```python
#데이터 로드
from sklearn import datasets
data = pd.read_csv('Auction_master_train.csv')
```


```python
data.columns
```




    Index(['Auction_key', 'Auction_class', 'Bid_class', 'Claim_price',
           'Appraisal_company', 'Appraisal_date', 'Auction_count',
           'Auction_miscarriage_count', 'Total_land_gross_area',
           'Total_land_real_area', 'Total_land_auction_area',
           'Total_building_area', 'Total_building_auction_area',
           'Total_appraisal_price', 'Minimum_sales_price', 'First_auction_date',
           'Final_auction_date', 'Final_result', 'Creditor', 'addr_do', 'addr_si',
           'addr_dong', 'addr_li', 'addr_san', 'addr_bunji1', 'addr_bunji2',
           'addr_etc', 'Apartment_usage', 'Preserve_regist_date', 'Total_floor',
           'Current_floor', 'Specific', 'Share_auction_YorN', 'road_name',
           'road_bunji1', 'road_bunji2', 'Close_date', 'Close_result', 'point.y',
           'point.x', 'Hammer_price'],
          dtype='object')




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
      <th>Auction_key</th>
      <th>Claim_price</th>
      <th>Auction_count</th>
      <th>Auction_miscarriage_count</th>
      <th>Total_land_gross_area</th>
      <th>Total_land_real_area</th>
      <th>Total_land_auction_area</th>
      <th>Total_building_area</th>
      <th>Total_building_auction_area</th>
      <th>Total_appraisal_price</th>
      <th>Minimum_sales_price</th>
      <th>addr_bunji1</th>
      <th>addr_bunji2</th>
      <th>Total_floor</th>
      <th>Current_floor</th>
      <th>road_bunji1</th>
      <th>road_bunji2</th>
      <th>point.y</th>
      <th>point.x</th>
      <th>Hammer_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1933.000000</td>
      <td>1.933000e+03</td>
      <td>1933.000000</td>
      <td>1933.000000</td>
      <td>1.933000e+03</td>
      <td>1933.000000</td>
      <td>1933.000000</td>
      <td>1933.000000</td>
      <td>1933.000000</td>
      <td>1.933000e+03</td>
      <td>1.933000e+03</td>
      <td>1929.000000</td>
      <td>889.000000</td>
      <td>1933.000000</td>
      <td>1933.000000</td>
      <td>1909.000000</td>
      <td>155.000000</td>
      <td>1933.000000</td>
      <td>1933.000000</td>
      <td>1.933000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1380.271081</td>
      <td>3.703908e+08</td>
      <td>1.836006</td>
      <td>0.788412</td>
      <td>3.458714e+04</td>
      <td>42.333802</td>
      <td>41.310776</td>
      <td>96.417693</td>
      <td>94.148810</td>
      <td>4.973592e+08</td>
      <td>4.155955e+08</td>
      <td>601.952307</td>
      <td>22.742407</td>
      <td>16.980859</td>
      <td>8.817900</td>
      <td>127.441069</td>
      <td>12.748387</td>
      <td>36.698018</td>
      <td>127.731667</td>
      <td>4.726901e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>801.670470</td>
      <td>1.337869e+09</td>
      <td>0.938319</td>
      <td>0.831715</td>
      <td>9.442101e+04</td>
      <td>65.274404</td>
      <td>65.385900</td>
      <td>106.323240</td>
      <td>106.845985</td>
      <td>7.873851e+08</td>
      <td>5.030312e+08</td>
      <td>554.119824</td>
      <td>67.000807</td>
      <td>9.509021</td>
      <td>8.044644</td>
      <td>188.394217</td>
      <td>10.735663</td>
      <td>1.150269</td>
      <td>0.993055</td>
      <td>5.574493e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.390000</td>
      <td>1.500000</td>
      <td>4.285000e+06</td>
      <td>4.285000e+06</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>35.051385</td>
      <td>126.809393</td>
      <td>6.303000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>691.000000</td>
      <td>7.746112e+07</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.997000e+03</td>
      <td>25.870000</td>
      <td>24.570000</td>
      <td>61.520000</td>
      <td>59.970000</td>
      <td>2.090000e+08</td>
      <td>1.750000e+08</td>
      <td>189.000000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>24.000000</td>
      <td>5.000000</td>
      <td>35.188590</td>
      <td>126.959167</td>
      <td>1.975550e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1395.000000</td>
      <td>1.728143e+08</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.424140e+04</td>
      <td>37.510000</td>
      <td>36.790000</td>
      <td>84.900000</td>
      <td>84.860000</td>
      <td>3.600000e+08</td>
      <td>3.120000e+08</td>
      <td>482.000000</td>
      <td>5.000000</td>
      <td>15.000000</td>
      <td>7.000000</td>
      <td>57.000000</td>
      <td>9.000000</td>
      <td>37.500862</td>
      <td>127.065003</td>
      <td>3.544500e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2062.000000</td>
      <td>3.565089e+08</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>4.140310e+04</td>
      <td>51.790000</td>
      <td>51.320000</td>
      <td>114.940000</td>
      <td>114.850000</td>
      <td>5.720000e+08</td>
      <td>4.864000e+08</td>
      <td>834.000000</td>
      <td>18.000000</td>
      <td>21.000000</td>
      <td>12.000000</td>
      <td>145.000000</td>
      <td>17.500000</td>
      <td>37.566116</td>
      <td>129.018054</td>
      <td>5.599000e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2762.000000</td>
      <td>2.286481e+10</td>
      <td>13.000000</td>
      <td>9.000000</td>
      <td>3.511936e+06</td>
      <td>2665.840000</td>
      <td>2665.840000</td>
      <td>4255.070000</td>
      <td>4255.070000</td>
      <td>2.777500e+10</td>
      <td>1.422080e+10</td>
      <td>4937.000000</td>
      <td>1414.000000</td>
      <td>80.000000</td>
      <td>65.000000</td>
      <td>1716.000000</td>
      <td>55.000000</td>
      <td>37.685575</td>
      <td>129.255872</td>
      <td>1.515100e+10</td>
    </tr>
  </tbody>
</table>
</div>



## EDA

- 'Auction_class'
- 'Bid_class'
- 'addr_do'
- 'Apartment_usage'
- 'Share_auction_YorN'
- 를 nominal data로 encoding 하겠습니다.


```python
data['Auction_class'].unique()
```




    array(['임의', '강제'], dtype=object)




```python
data['Auction_class']=data['Auction_class'].replace(['임의', '강제'],[0,1])
```


```python
data['Bid_class'].unique()
```




    array(['개별', '일반', '일괄'], dtype=object)




```python
data['Bid_class']=data['Bid_class'].replace(['개별', '일반', '일괄'],[0,1,2])
```


```python
data['addr_do'].unique()
```




    array(['부산', '서울'], dtype=object)




```python
data['addr_do']=data['addr_do'].replace(['부산','서울'],[0,1])
```


```python
data['Apartment_usage'].unique()
```




    array(['주상복합', '아파트'], dtype=object)




```python
data['Apartment_usage']=data['Apartment_usage'].replace(['주상복합', '아파트'],[0,1])
```


```python
data['Share_auction_YorN'].unique()
```




    array(['N', 'Y'], dtype=object)




```python
data['Share_auction_YorN']=data['Share_auction_YorN'].replace(['N', 'Y'],[0,1])
```


```python
col_np=np.array(['Appraisal_date',
'First_auction_date',
'Final_auction_date',
'Preserve_regist_date',
'Close_date'])
```


```python
for index, row in data[col_np].iterrows():
    for i in col_np:
        val= row[i].split('-')
        last=val[2].split(':')
        del val[2]
        val.append(last[0].split(' ')[0])
        data.loc[index,i]=''.join(val)
```


```python
for i in col_np:
    data[i]=data[i].astype(int)
```

Ref : https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

## 변수 제거, 선택 

- 'Auction_key'와 같이 이산적/연속적으로 분포하더라도, data 값이 별다른 의미를 함축하지 않거나, 
- 'Auction_class'와 같이 type이 string인 범주형 변수
- 를 위주로 제거했습니다.


```python
data = data.drop(['Auction_key', 'Appraisal_company', 'Creditor', 'addr_si',
       'addr_dong', 'addr_li', 'addr_san', 'addr_bunji1', 'addr_bunji2',
       'addr_etc', 'Specific', 'road_name',
       'road_bunji1', 'road_bunji2' , 'Final_result', 'Close_result'], axis=1)
```


```python
data. columns
```




    Index(['Auction_class', 'Bid_class', 'Claim_price', 'Appraisal_date',
           'Auction_count', 'Auction_miscarriage_count', 'Total_land_gross_area',
           'Total_land_real_area', 'Total_land_auction_area',
           'Total_building_area', 'Total_building_auction_area',
           'Total_appraisal_price', 'Minimum_sales_price', 'First_auction_date',
           'Final_auction_date', 'addr_do', 'Apartment_usage',
           'Preserve_regist_date', 'Total_floor', 'Current_floor',
           'Share_auction_YorN', 'Close_date', 'point.y', 'point.x',
           'Hammer_price'],
          dtype='object')



## 다중 공선성 확인 

- 이 data set에서는 target data가 'Hammer_price'로 뚜렷하므로, df_x, df_y로 변수를 구분 짓겠습니다.


```python
data_org_x = data.drop(['Hammer_price'], axis=1) # 공선성을 갖는 feature를 삭제하지 않을, 대조군 데이터입니다.
data_x = data.drop(['Hammer_price'], axis=1) #나중에 VIF를 이용해서 처리할, VIF 데이터라고 명명하겠습니다.
data_y = pd.DataFrame(data['Hammer_price'],columns=['Hammer_price'])
```


```python
# #산점도 행렬
# sns_plot = sns.pairplot(data_x)
# sns_plot.savefig("output.png")
```

- 주피터 상으로는 column이 잘 안보여서, 이미지를 추출해서 확인했습니다.
- 코드 출처 : https://stackoverflow.com/questions/32244753/how-to-save-a-seaborn-plot-into-a-file

1. 'Auction_count' / 'Auction_miscarriage_count'
2. 'Total_land_real_area' / 'Total_land_auction_area' /  'Total_building_area' /  'Total_building_auction_area'
3. 'Total_appraisal_price' / 'Minimum_sales_price' / 
4. 'Total_floor' / 'Current_floor'
5. 'point.x' / 'point.y'

- 다음 5개의 그룹에 속한 변수들끼리는 상관관계가 있는것을 확인할 수 있습니다.
- 그리고 2번째 그룹과 3번째 그룹에 속한 변수들끼리도 상관관계가 있습니다.
- 5번째 그룹에 속한 두개의 변수 사이에는 음의 상관관계가 있습니다.
- 서울의 대략적인 좌표는 (127,37), 부산의 대략적인 좌표는 (129, 35) 입니다.
- point.x가 커지면 부산 지역이니까, point.y값이 작아지고, point.x값이 작아지면 서울지역이니까 point.y값이 커집니다.


```python
#상관계수 행렬
data_x.corr()
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
      <th>Auction_class</th>
      <th>Bid_class</th>
      <th>Claim_price</th>
      <th>Appraisal_date</th>
      <th>Auction_count</th>
      <th>Auction_miscarriage_count</th>
      <th>Total_land_gross_area</th>
      <th>Total_land_real_area</th>
      <th>Total_land_auction_area</th>
      <th>Total_building_area</th>
      <th>...</th>
      <th>Final_auction_date</th>
      <th>addr_do</th>
      <th>Apartment_usage</th>
      <th>Preserve_regist_date</th>
      <th>Total_floor</th>
      <th>Current_floor</th>
      <th>Share_auction_YorN</th>
      <th>Close_date</th>
      <th>point.y</th>
      <th>point.x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Auction_class</th>
      <td>1.000000</td>
      <td>-0.128420</td>
      <td>-0.052025</td>
      <td>-0.044855</td>
      <td>0.064498</td>
      <td>0.067789</td>
      <td>-0.045935</td>
      <td>-0.005182</td>
      <td>-0.021547</td>
      <td>-0.013988</td>
      <td>...</td>
      <td>-0.009420</td>
      <td>-0.033180</td>
      <td>-0.132727</td>
      <td>-0.112866</td>
      <td>-0.090146</td>
      <td>-0.015933</td>
      <td>0.208536</td>
      <td>-0.014139</td>
      <td>-0.031377</td>
      <td>0.034449</td>
    </tr>
    <tr>
      <th>Bid_class</th>
      <td>-0.128420</td>
      <td>1.000000</td>
      <td>-0.233687</td>
      <td>0.232062</td>
      <td>-0.151884</td>
      <td>-0.160280</td>
      <td>0.055316</td>
      <td>0.161695</td>
      <td>0.158624</td>
      <td>0.148806</td>
      <td>...</td>
      <td>0.058097</td>
      <td>0.002516</td>
      <td>0.226296</td>
      <td>0.169129</td>
      <td>0.100746</td>
      <td>0.019175</td>
      <td>0.026340</td>
      <td>0.007530</td>
      <td>0.000615</td>
      <td>0.003108</td>
    </tr>
    <tr>
      <th>Claim_price</th>
      <td>-0.052025</td>
      <td>-0.233687</td>
      <td>1.000000</td>
      <td>-0.154947</td>
      <td>0.013312</td>
      <td>0.015411</td>
      <td>0.003771</td>
      <td>0.180421</td>
      <td>0.182207</td>
      <td>0.202379</td>
      <td>...</td>
      <td>-0.015191</td>
      <td>0.100203</td>
      <td>0.006534</td>
      <td>-0.045835</td>
      <td>0.016466</td>
      <td>0.043258</td>
      <td>-0.029882</td>
      <td>0.004733</td>
      <td>0.095939</td>
      <td>-0.103605</td>
    </tr>
    <tr>
      <th>Appraisal_date</th>
      <td>-0.044855</td>
      <td>0.232062</td>
      <td>-0.154947</td>
      <td>1.000000</td>
      <td>-0.290210</td>
      <td>-0.272819</td>
      <td>0.021961</td>
      <td>-0.014571</td>
      <td>-0.019146</td>
      <td>-0.042595</td>
      <td>...</td>
      <td>0.614390</td>
      <td>-0.097096</td>
      <td>0.124106</td>
      <td>0.274513</td>
      <td>0.056611</td>
      <td>-0.003312</td>
      <td>0.055550</td>
      <td>-0.042243</td>
      <td>-0.097088</td>
      <td>0.099583</td>
    </tr>
    <tr>
      <th>Auction_count</th>
      <td>0.064498</td>
      <td>-0.151884</td>
      <td>0.013312</td>
      <td>-0.290210</td>
      <td>1.000000</td>
      <td>0.972918</td>
      <td>-0.045697</td>
      <td>0.062824</td>
      <td>0.062868</td>
      <td>0.107074</td>
      <td>...</td>
      <td>-0.001544</td>
      <td>-0.053300</td>
      <td>-0.112422</td>
      <td>-0.276032</td>
      <td>-0.052329</td>
      <td>-0.029124</td>
      <td>0.019374</td>
      <td>-0.006374</td>
      <td>-0.054074</td>
      <td>0.051634</td>
    </tr>
    <tr>
      <th>Auction_miscarriage_count</th>
      <td>0.067789</td>
      <td>-0.160280</td>
      <td>0.015411</td>
      <td>-0.272819</td>
      <td>0.972918</td>
      <td>1.000000</td>
      <td>-0.050457</td>
      <td>0.069582</td>
      <td>0.069392</td>
      <td>0.118290</td>
      <td>...</td>
      <td>0.003906</td>
      <td>-0.056091</td>
      <td>-0.121830</td>
      <td>-0.295186</td>
      <td>-0.045343</td>
      <td>-0.021852</td>
      <td>0.024246</td>
      <td>-0.011688</td>
      <td>-0.057197</td>
      <td>0.054032</td>
    </tr>
    <tr>
      <th>Total_land_gross_area</th>
      <td>-0.045935</td>
      <td>0.055316</td>
      <td>0.003771</td>
      <td>0.021961</td>
      <td>-0.045697</td>
      <td>-0.050457</td>
      <td>1.000000</td>
      <td>0.049791</td>
      <td>0.048225</td>
      <td>0.017401</td>
      <td>...</td>
      <td>0.006334</td>
      <td>0.016506</td>
      <td>0.130701</td>
      <td>0.049272</td>
      <td>0.100557</td>
      <td>0.094385</td>
      <td>-0.000546</td>
      <td>0.010173</td>
      <td>0.014971</td>
      <td>-0.014285</td>
    </tr>
    <tr>
      <th>Total_land_real_area</th>
      <td>-0.005182</td>
      <td>0.161695</td>
      <td>0.180421</td>
      <td>-0.014571</td>
      <td>0.062824</td>
      <td>0.069582</td>
      <td>0.049791</td>
      <td>1.000000</td>
      <td>0.996224</td>
      <td>0.940361</td>
      <td>...</td>
      <td>-0.013894</td>
      <td>0.064039</td>
      <td>0.120876</td>
      <td>-0.148100</td>
      <td>-0.072543</td>
      <td>-0.062673</td>
      <td>-0.003757</td>
      <td>-0.004496</td>
      <td>0.060602</td>
      <td>-0.062621</td>
    </tr>
    <tr>
      <th>Total_land_auction_area</th>
      <td>-0.021547</td>
      <td>0.158624</td>
      <td>0.182207</td>
      <td>-0.019146</td>
      <td>0.062868</td>
      <td>0.069392</td>
      <td>0.048225</td>
      <td>0.996224</td>
      <td>1.000000</td>
      <td>0.938144</td>
      <td>...</td>
      <td>-0.018264</td>
      <td>0.059051</td>
      <td>0.117132</td>
      <td>-0.148266</td>
      <td>-0.068385</td>
      <td>-0.061274</td>
      <td>-0.079567</td>
      <td>-0.004583</td>
      <td>0.055776</td>
      <td>-0.057796</td>
    </tr>
    <tr>
      <th>Total_building_area</th>
      <td>-0.013988</td>
      <td>0.148806</td>
      <td>0.202379</td>
      <td>-0.042595</td>
      <td>0.107074</td>
      <td>0.118290</td>
      <td>0.017401</td>
      <td>0.940361</td>
      <td>0.938144</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.022298</td>
      <td>0.067411</td>
      <td>-0.000785</td>
      <td>-0.186586</td>
      <td>0.107069</td>
      <td>0.071110</td>
      <td>-0.010387</td>
      <td>0.002245</td>
      <td>0.062797</td>
      <td>-0.064572</td>
    </tr>
    <tr>
      <th>Total_building_auction_area</th>
      <td>-0.035694</td>
      <td>0.144568</td>
      <td>0.204000</td>
      <td>-0.048350</td>
      <td>0.104286</td>
      <td>0.114779</td>
      <td>0.016474</td>
      <td>0.934759</td>
      <td>0.941681</td>
      <td>0.993533</td>
      <td>...</td>
      <td>-0.028215</td>
      <td>0.060118</td>
      <td>-0.002579</td>
      <td>-0.186185</td>
      <td>0.108707</td>
      <td>0.069819</td>
      <td>-0.113234</td>
      <td>0.002390</td>
      <td>0.055730</td>
      <td>-0.057502</td>
    </tr>
    <tr>
      <th>Total_appraisal_price</th>
      <td>-0.031005</td>
      <td>0.138715</td>
      <td>0.248846</td>
      <td>-0.040338</td>
      <td>0.071038</td>
      <td>0.077547</td>
      <td>0.041896</td>
      <td>0.842248</td>
      <td>0.845243</td>
      <td>0.900302</td>
      <td>...</td>
      <td>-0.013409</td>
      <td>0.229401</td>
      <td>-0.009252</td>
      <td>-0.125375</td>
      <td>0.169577</td>
      <td>0.151757</td>
      <td>-0.067888</td>
      <td>0.000467</td>
      <td>0.220942</td>
      <td>-0.222110</td>
    </tr>
    <tr>
      <th>Minimum_sales_price</th>
      <td>-0.054143</td>
      <td>0.143433</td>
      <td>0.261076</td>
      <td>-0.020236</td>
      <td>-0.036716</td>
      <td>-0.033039</td>
      <td>0.067135</td>
      <td>0.711515</td>
      <td>0.715891</td>
      <td>0.783299</td>
      <td>...</td>
      <td>-0.004068</td>
      <td>0.295578</td>
      <td>-0.002602</td>
      <td>-0.076131</td>
      <td>0.223407</td>
      <td>0.201913</td>
      <td>-0.088098</td>
      <td>0.000337</td>
      <td>0.285141</td>
      <td>-0.285804</td>
    </tr>
    <tr>
      <th>First_auction_date</th>
      <td>-0.032737</td>
      <td>0.150110</td>
      <td>-0.029397</td>
      <td>0.761737</td>
      <td>-0.325041</td>
      <td>-0.315996</td>
      <td>0.028743</td>
      <td>-0.012214</td>
      <td>-0.017502</td>
      <td>-0.041639</td>
      <td>...</td>
      <td>0.756500</td>
      <td>-0.035875</td>
      <td>0.100524</td>
      <td>0.260852</td>
      <td>0.024827</td>
      <td>-0.012017</td>
      <td>0.056159</td>
      <td>0.006194</td>
      <td>-0.036108</td>
      <td>0.036436</td>
    </tr>
    <tr>
      <th>Final_auction_date</th>
      <td>-0.009420</td>
      <td>0.058097</td>
      <td>-0.015191</td>
      <td>0.614390</td>
      <td>-0.001544</td>
      <td>0.003906</td>
      <td>0.006334</td>
      <td>-0.013894</td>
      <td>-0.018264</td>
      <td>-0.022298</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.025143</td>
      <td>0.035163</td>
      <td>0.089253</td>
      <td>0.014742</td>
      <td>-0.017708</td>
      <td>0.053887</td>
      <td>-0.049619</td>
      <td>-0.025539</td>
      <td>0.024945</td>
    </tr>
    <tr>
      <th>addr_do</th>
      <td>-0.033180</td>
      <td>0.002516</td>
      <td>0.100203</td>
      <td>-0.097096</td>
      <td>-0.053300</td>
      <td>-0.056091</td>
      <td>0.016506</td>
      <td>0.064039</td>
      <td>0.059051</td>
      <td>0.067411</td>
      <td>...</td>
      <td>-0.025143</td>
      <td>1.000000</td>
      <td>0.080037</td>
      <td>0.054499</td>
      <td>-0.165918</td>
      <td>-0.126812</td>
      <td>0.050380</td>
      <td>-0.059284</td>
      <td>0.998749</td>
      <td>-0.996850</td>
    </tr>
    <tr>
      <th>Apartment_usage</th>
      <td>-0.132727</td>
      <td>0.226296</td>
      <td>0.006534</td>
      <td>0.124106</td>
      <td>-0.112422</td>
      <td>-0.121830</td>
      <td>0.130701</td>
      <td>0.120876</td>
      <td>0.117132</td>
      <td>-0.000785</td>
      <td>...</td>
      <td>0.035163</td>
      <td>0.080037</td>
      <td>1.000000</td>
      <td>0.137427</td>
      <td>-0.098981</td>
      <td>-0.132445</td>
      <td>0.024764</td>
      <td>-0.013297</td>
      <td>0.081952</td>
      <td>-0.079041</td>
    </tr>
    <tr>
      <th>Preserve_regist_date</th>
      <td>-0.112866</td>
      <td>0.169129</td>
      <td>-0.045835</td>
      <td>0.274513</td>
      <td>-0.276032</td>
      <td>-0.295186</td>
      <td>0.049272</td>
      <td>-0.148100</td>
      <td>-0.148266</td>
      <td>-0.186586</td>
      <td>...</td>
      <td>0.089253</td>
      <td>0.054499</td>
      <td>0.137427</td>
      <td>1.000000</td>
      <td>0.057654</td>
      <td>0.030474</td>
      <td>0.006047</td>
      <td>0.032180</td>
      <td>0.052926</td>
      <td>-0.056815</td>
    </tr>
    <tr>
      <th>Total_floor</th>
      <td>-0.090146</td>
      <td>0.100746</td>
      <td>0.016466</td>
      <td>0.056611</td>
      <td>-0.052329</td>
      <td>-0.045343</td>
      <td>0.100557</td>
      <td>-0.072543</td>
      <td>-0.068385</td>
      <td>0.107069</td>
      <td>...</td>
      <td>0.014742</td>
      <td>-0.165918</td>
      <td>-0.098981</td>
      <td>0.057654</td>
      <td>1.000000</td>
      <td>0.708154</td>
      <td>-0.052357</td>
      <td>0.033836</td>
      <td>-0.167823</td>
      <td>0.172578</td>
    </tr>
    <tr>
      <th>Current_floor</th>
      <td>-0.015933</td>
      <td>0.019175</td>
      <td>0.043258</td>
      <td>-0.003312</td>
      <td>-0.029124</td>
      <td>-0.021852</td>
      <td>0.094385</td>
      <td>-0.062673</td>
      <td>-0.061274</td>
      <td>0.071110</td>
      <td>...</td>
      <td>-0.017708</td>
      <td>-0.126812</td>
      <td>-0.132445</td>
      <td>0.030474</td>
      <td>0.708154</td>
      <td>1.000000</td>
      <td>-0.016769</td>
      <td>0.028320</td>
      <td>-0.128727</td>
      <td>0.131782</td>
    </tr>
    <tr>
      <th>Share_auction_YorN</th>
      <td>0.208536</td>
      <td>0.026340</td>
      <td>-0.029882</td>
      <td>0.055550</td>
      <td>0.019374</td>
      <td>0.024246</td>
      <td>-0.000546</td>
      <td>-0.003757</td>
      <td>-0.079567</td>
      <td>-0.010387</td>
      <td>...</td>
      <td>0.053887</td>
      <td>0.050380</td>
      <td>0.024764</td>
      <td>0.006047</td>
      <td>-0.052357</td>
      <td>-0.016769</td>
      <td>1.000000</td>
      <td>-0.016363</td>
      <td>0.049825</td>
      <td>-0.049341</td>
    </tr>
    <tr>
      <th>Close_date</th>
      <td>-0.014139</td>
      <td>0.007530</td>
      <td>0.004733</td>
      <td>-0.042243</td>
      <td>-0.006374</td>
      <td>-0.011688</td>
      <td>0.010173</td>
      <td>-0.004496</td>
      <td>-0.004583</td>
      <td>0.002245</td>
      <td>...</td>
      <td>-0.049619</td>
      <td>-0.059284</td>
      <td>-0.013297</td>
      <td>0.032180</td>
      <td>0.033836</td>
      <td>0.028320</td>
      <td>-0.016363</td>
      <td>1.000000</td>
      <td>-0.059843</td>
      <td>0.057197</td>
    </tr>
    <tr>
      <th>point.y</th>
      <td>-0.031377</td>
      <td>0.000615</td>
      <td>0.095939</td>
      <td>-0.097088</td>
      <td>-0.054074</td>
      <td>-0.057197</td>
      <td>0.014971</td>
      <td>0.060602</td>
      <td>0.055776</td>
      <td>0.062797</td>
      <td>...</td>
      <td>-0.025539</td>
      <td>0.998749</td>
      <td>0.081952</td>
      <td>0.052926</td>
      <td>-0.167823</td>
      <td>-0.128727</td>
      <td>0.049825</td>
      <td>-0.059843</td>
      <td>1.000000</td>
      <td>-0.994182</td>
    </tr>
    <tr>
      <th>point.x</th>
      <td>0.034449</td>
      <td>0.003108</td>
      <td>-0.103605</td>
      <td>0.099583</td>
      <td>0.051634</td>
      <td>0.054032</td>
      <td>-0.014285</td>
      <td>-0.062621</td>
      <td>-0.057796</td>
      <td>-0.064572</td>
      <td>...</td>
      <td>0.024945</td>
      <td>-0.996850</td>
      <td>-0.079041</td>
      <td>-0.056815</td>
      <td>0.172578</td>
      <td>0.131782</td>
      <td>-0.049341</td>
      <td>0.057197</td>
      <td>-0.994182</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 24 columns</p>
</div>




```python
#상관계수 행렬
plt.figure(figsize=(10,10))
sns.heatmap(data = data_x.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc28ceb0>




![png](week2_regression_wk3_%EA%B8%B8%ED%83%9C%ED%98%95_files/week2_regression_wk3_%EA%B8%B8%ED%83%9C%ED%98%95_32_1.png)


- 상관계수로 분석해도 비슷한 결과를 얻을 수 있습니다.
- 산점도 행렬을 분석했을때, 같은 그룹으로 묶였던 변수들끼리는 절대값이 큰 상관계수 값을 갖습니다. 


```python
#VIF확인하기
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_x.values, i) for i in range(data_x.shape[1])]
vif["features"] = data_x.columns
vif.sort_values(["VIF Factor"], ascending=[False])
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>998.916653</td>
      <td>Total_land_auction_area</td>
    </tr>
    <tr>
      <th>7</th>
      <td>980.221095</td>
      <td>Total_land_real_area</td>
    </tr>
    <tr>
      <th>15</th>
      <td>970.527823</td>
      <td>addr_do</td>
    </tr>
    <tr>
      <th>10</th>
      <td>787.078620</td>
      <td>Total_building_auction_area</td>
    </tr>
    <tr>
      <th>9</th>
      <td>761.539195</td>
      <td>Total_building_area</td>
    </tr>
    <tr>
      <th>22</th>
      <td>530.756014</td>
      <td>point.y</td>
    </tr>
    <tr>
      <th>23</th>
      <td>164.024599</td>
      <td>point.x</td>
    </tr>
    <tr>
      <th>11</th>
      <td>68.460629</td>
      <td>Total_appraisal_price</td>
    </tr>
    <tr>
      <th>12</th>
      <td>36.043627</td>
      <td>Minimum_sales_price</td>
    </tr>
    <tr>
      <th>5</th>
      <td>19.557042</td>
      <td>Auction_miscarriage_count</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.384641</td>
      <td>Auction_count</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6.024271</td>
      <td>Share_auction_YorN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4.205747</td>
      <td>First_auction_date</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.682289</td>
      <td>Appraisal_date</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.629700</td>
      <td>Total_floor</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.513257</td>
      <td>Final_auction_date</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.087449</td>
      <td>Current_floor</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.297487</td>
      <td>Apartment_usage</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.284034</td>
      <td>Bid_class</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.257005</td>
      <td>Preserve_regist_date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.219651</td>
      <td>Claim_price</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.111321</td>
      <td>Auction_class</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.068419</td>
      <td>Total_land_gross_area</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.016970</td>
      <td>Close_date</td>
    </tr>
  </tbody>
</table>
</div>



- VIF가 10넘어가는 값이 
- Total_land_auction_area
- Total_land_real_area
- addr_do
- Total_building_auction_area
- Total_building_area
- point.y
- point.x
- Total_appraisal_price
- Minimum_sales_price
- Auction_miscarriage_count
- Auction_count

- 이렇게 많고, VIF값도 10보다 많이 큽니다...

- 집의 면적, 경매 횟수, 판매 가격에 관련된 변수들이 중복해서 여럿 등장하는 것을 알 수 있습니다.
- 각 데이터를 대표할 수 있는 변수만 남기고, 나머지는 삭제하겠습니다.


```python
data_x = data_x.drop(['point.x','point.y','Total_land_auction_area','Total_land_real_area','Total_building_auction_area','Total_building_auction_area','Minimum_sales_price','Auction_miscarriage_count'],axis=1)
```


```python
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_x.values, i) for i in range(data_x.shape[1])]
vif["features"] = data_x.columns
vif.sort_values(["VIF Factor"], ascending=[False])
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>7.176322</td>
      <td>Total_appraisal_price</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.575172</td>
      <td>Total_building_area</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.154124</td>
      <td>First_auction_date</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.643818</td>
      <td>Appraisal_date</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.391308</td>
      <td>Final_auction_date</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.131880</td>
      <td>Total_floor</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.081988</td>
      <td>Current_floor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.402001</td>
      <td>Auction_count</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.335113</td>
      <td>addr_do</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.273709</td>
      <td>Bid_class</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.221025</td>
      <td>Preserve_regist_date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.209336</td>
      <td>Claim_price</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.138629</td>
      <td>Apartment_usage</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.106482</td>
      <td>Auction_class</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.088672</td>
      <td>Share_auction_YorN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.036720</td>
      <td>Total_land_gross_area</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.015179</td>
      <td>Close_date</td>
    </tr>
  </tbody>
</table>
</div>



- 모든 feature의 VIF값이 10보다 작아지도록 하는데 성공했습니다!

## Modeling

### R-Square, Adj. R-square 값 구하기


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0)
```


```python
from sklearn.linear_model import LinearRegression

#모델 불러옴
model = LinearRegression()
#train data에 fit시킴
model.fit(X_train, y_train)
model.predict(X_train)
```




    array([[2.42153261e+08],
           [3.90730591e+08],
           [5.14365307e+08],
           ...,
           [5.30745753e+08],
           [1.87207017e+08],
           [1.75620882e+08]])




```python
#fit된 모델의 R-square
model.score(X_train, y_train)
```




    0.9632993924275883



- 더 정확한 값을 얻기 위해 OLS Model을 이용하겠습니다. 


```python
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

model1=sm.OLS(y_train,X_train)
result=model1.fit()
print(result.summary())
# Ref : https://discuss.analyticsvidhya.com/t/getting-p-value-r-squared-and-adjusted-r-squared-value-in-python/31528
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           Hammer_price   R-squared:                       0.963
    Model:                            OLS   Adj. R-squared:                  0.963
    Method:                 Least Squares   F-statistic:                     2508.
    Date:                Sat, 03 Aug 2019   Prob (F-statistic):               0.00
    Time:                        15:52:07   Log-Likelihood:                -30877.
    No. Observations:                1546   AIC:                         6.179e+04
    Df Residuals:                    1529   BIC:                         6.188e+04
    Df Model:                          16                                         
    Covariance Type:            nonrobust                                         
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Auction_class         -8.366e+06   7.08e+06     -1.182      0.237   -2.22e+07    5.51e+06
    Bid_class              3.812e+07   1.27e+07      3.000      0.003    1.32e+07    6.31e+07
    Claim_price               0.0107      0.003      3.808      0.000       0.005       0.016
    Appraisal_date         -503.4820    529.370     -0.951      0.342   -1541.850     534.886
    Auction_count         -3.752e+07   3.65e+06    -10.285      0.000   -4.47e+07   -3.04e+07
    Total_land_gross_area    73.7470     28.666      2.573      0.010      17.519     129.975
    Total_building_area    -2.34e+06   6.72e+04    -34.835      0.000   -2.47e+06   -2.21e+06
    Total_appraisal_price     0.9397      0.010     98.040      0.000       0.921       0.958
    First_auction_date    -1459.2689    760.307     -1.919      0.055   -2950.623      32.086
    Final_auction_date     1971.7164    675.669      2.918      0.004     646.381    3297.052
    addr_do                3.606e+07   7.01e+06      5.145      0.000    2.23e+07    4.98e+07
    Apartment_usage        1.928e+07   9.07e+06      2.125      0.034    1.48e+06    3.71e+07
    Preserve_regist_date     -3.8494      2.442     -1.577      0.115      -8.639       0.940
    Total_floor             2.57e+06   4.56e+05      5.633      0.000    1.67e+06    3.46e+06
    Current_floor         -1.442e+05   5.34e+05     -0.270      0.787   -1.19e+06    9.02e+05
    Share_auction_YorN    -7.757e+06   1.46e+07     -0.530      0.596   -3.65e+07    2.09e+07
    Close_date                3.4021      4.283      0.794      0.427      -5.000      11.804
    ==============================================================================
    Omnibus:                      469.832   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            25009.015
    Skew:                           0.594   Prob(JB):                         0.00
    Kurtosis:                      22.668   Cond. No.                     6.71e+09
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 6.71e+09. This might indicate that there are
    strong multicollinearity or other numerical problems.


- VIF값을 참조해서 Feature를 삭제하지 않은, 대조군 데이터와 비교해보겠습니다.


```python
#대조군과 비교.
X_train2, X_test2, y_train2, y_test2 = train_test_split(data_org_x, data_y, test_size=0.2, random_state=0)

#모델 불러옴
model = LinearRegression()
#train data에 fit시킴
model.fit(X_train2, y_train2)
#fit된 모델의 R-square
model.score(X_train2, y_train2)
```




    0.9913076440626994



- Feature를 추가적으로 삭제하지 않은 대조군 데이터의 R-square값이 더 큽니다
- 대조군 데이터의 feature의 갯수가 더 많기 때문인것으로 예상해볼수 있습니다.
- Adjusted R-squared는 다를 수 있겠죠


```python
model1=sm.OLS(y_train2,X_train2)
result=model1.fit()
print(result.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           Hammer_price   R-squared:                       0.991
    Model:                            OLS   Adj. R-squared:                  0.991
    Method:                 Least Squares   F-statistic:                     7536.
    Date:                Sat, 03 Aug 2019   Prob (F-statistic):               0.00
    Time:                        15:52:07   Log-Likelihood:                -29765.
    No. Observations:                1546   AIC:                         5.958e+04
    Df Residuals:                    1522   BIC:                         5.971e+04
    Df Model:                          23                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================================
                                      coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------------
    Auction_class                2.841e+06   3.46e+06      0.821      0.412   -3.95e+06    9.63e+06
    Bid_class                    2.189e+07   6.22e+06      3.518      0.000    9.69e+06    3.41e+07
    Claim_price                     0.0039      0.001      2.857      0.004       0.001       0.007
    Appraisal_date               -360.5193    260.610     -1.383      0.167    -871.712     150.673
    Auction_count               -1.899e+07    6.4e+06     -2.966      0.003   -3.15e+07   -6.43e+06
    Auction_miscarriage_count    3.832e+07   7.33e+06      5.230      0.000     2.4e+07    5.27e+07
    Total_land_gross_area          21.2514     14.192      1.497      0.134      -6.587      49.090
    Total_land_real_area         6.772e+05   6.29e+05      1.076      0.282   -5.57e+05    1.91e+06
    Total_land_auction_area     -9.189e+05   6.35e+05     -1.446      0.148   -2.17e+06    3.28e+05
    Total_building_area         -2.088e+05   3.38e+05     -0.618      0.537   -8.72e+05    4.54e+05
    Total_building_auction_area  1.209e+04   3.43e+05      0.035      0.972    -6.6e+05    6.84e+05
    Total_appraisal_price           0.0607      0.015      4.110      0.000       0.032       0.090
    Minimum_sales_price             1.0569      0.017     63.138      0.000       1.024       1.090
    First_auction_date            487.7158    373.129      1.307      0.191    -244.185    1219.617
    Final_auction_date           -316.4510    338.583     -0.935      0.350    -980.589     347.687
    addr_do                       3.56e+08   9.23e+07      3.858      0.000    1.75e+08    5.37e+08
    Apartment_usage              1.812e+07   4.68e+06      3.870      0.000    8.94e+06    2.73e+07
    Preserve_regist_date            0.8655      1.210      0.716      0.474      -1.507       3.238
    Total_floor                 -7978.7022   2.46e+05     -0.032      0.974   -4.91e+05    4.75e+05
    Current_floor               -3.803e+05   2.61e+05     -1.456      0.146   -8.93e+05    1.32e+05
    Share_auction_YorN            -3.4e+07   1.67e+07     -2.032      0.042   -6.68e+07   -1.18e+06
    Close_date                      0.0097      2.093      0.005      0.996      -4.096       4.115
    point.y                     -9.594e+07   2.83e+07     -3.394      0.001   -1.51e+08   -4.05e+07
    point.x                      5.553e+07   1.85e+07      3.004      0.003    1.93e+07    9.18e+07
    ==============================================================================
    Omnibus:                     1122.785   Durbin-Watson:                   2.015
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            41986.317
    Skew:                           2.942   Prob(JB):                         0.00
    Kurtosis:                      27.843   Cond. No.                     9.69e+10
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 9.69e+10. This might indicate that there are
    strong multicollinearity or other numerical problems.


- Adjusted R.square역시 대조군 데이터가 더 큽니다. 
- 이유는 잘 모르겠습니다... 

### MSE값 구하기


```python
#MSE _ VIF 값을 이용해서 Feature를 처리한 데이터
import sklearn as sk
model = LinearRegression()
#train data에 fit시킴
model.fit(X_train, y_train)
#fit된 모델의 R-square
sk.metrics.mean_squared_error(y_train, model.predict(X_train))
```




    1.3034699472275636e+16




```python
#MSE _ 대조군 데이터
#train data에 fit시킴
model.fit(X_train2, y_train2)
#fit된 모델의 R-square
sk.metrics.mean_squared_error(y_train, model.predict(X_train2))
```




    3087203587167156.5



- 두 데이터군 모두 MSE가 비정상적으로 큰 것 같습니다...
- MSE값 역시, 대조군 데이터가 작습니다.
- 제가 Feature 삭제를 잘못한것 같습니다...


```python
#VIF 데이터
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)
```




    0.9193927539165223




```python
#대조군 데이터
model.fit(X_train2,y_train2)
model.predict(X_test2)
model.score(X_test2,y_test2)
```




    0.9851442258074934



- test Date일때보다, R-square이 두 데이터군 모두 감소했습니다.
- 대조군 데이터의 성능이 여전히 좋게 나옵니다.


```python
# 예측 vs. 실제데이터 plot - VIF 데이터
model.fit(X_train,y_train)
y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')

# 예측과 실제가 비슷하면, 라인상에 분포함
x = np.linspace(0, 3000000000, 100)
y = x
plt.plot(x, y)
plt.show()
```


![png](week2_regression_wk3_%EA%B8%B8%ED%83%9C%ED%98%95_files/week2_regression_wk3_%EA%B8%B8%ED%83%9C%ED%98%95_60_0.png)



```python
# 예측 vs. 실제데이터 plot - 대조군 데이터
model.fit(X_train2,y_train2)
y_pred2 = model.predict(X_test2) 
plt.plot(y_test2, y_pred2, '.')

# 예측과 실제가 비슷하면, 라인상에 분포함
x = np.linspace(0, 3000000000, 100)
y = x
plt.plot(x, y)
plt.show()
```


![png](week2_regression_wk3_%EA%B8%B8%ED%83%9C%ED%98%95_files/week2_regression_wk3_%EA%B8%B8%ED%83%9C%ED%98%95_61_0.png)


- TEST data에서도 대조군 데이터의 성능이 더 우수합니다.


```python
"""
============VIF 데이터군 기준============
    MSE: 1.3034699472275636e+16
    train R-square: 0.963
    test R-square: 0.919
=========================================
============대조군 데이터군 기준|===========
    MSE: 3087203587167156.5
    train R-square: 0.991
    test R-square: 0.985
=========================================
"""
```




    '\n============VIF 데이터군 기준============\n    MSE: 1.3034699472275636e+16\n    train R-square: 0.963\n    test R-square: 0.919\n=========================================\n============대조군 데이터군 기준|===========\n    MSE: 3087203587167156.5\n    train R-square: 0.991\n    test R-square: 0.985\n=========================================\n'



## 정규화


```python
#Ridge, Lasso 회귀
from sklearn.linear_model import Ridge, Lasso

ridge=Ridge(alpha=1.0)#alpha: 얼마나 정규화를 할건지 정하는 양수 하이퍼파라미터 (클수록 더 정규화)
ridge.fit(X_train, y_train)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=2.00753e-20): result may not be accurate.
      overwrite_a=True).T





    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)




```python
ridge.get_params
```




    <bound method BaseEstimator.get_params of Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)>




```python
#R-square
ridge.score(X_train,y_train)
```




    0.9632993499297073




```python
ridge.score(X_test,y_test)
```




    0.9193785506772447




```python
#정규화를 덜하니까 R-square가 아주 조금 증가했습니다.
ridge=Ridge(alpha=0.3)
ridge.fit(X_train, y_train)
ridge.score(X_train,y_train)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.98312e-20): result may not be accurate.
      overwrite_a=True).T





    0.9632993885502573




```python
ridge.score(X_test,y_test)
```




    0.9193884887331985




```python
#Lasso
lasso=Lasso(alpha=0.3)
lasso.fit(X_train, y_train)
lasso.score(X_train, y_train)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\coordinate_descent.py:475: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 9.796072269525045e+18, tolerance: 5.490820647690414e+16
      positive)





    0.9632993924275884




```python
lasso.score(X_test, y_test)
```




    0.9193927512317135



#  대조군 데이터로 해보기.


```python
ridge=Ridge(alpha=1)
ridge.fit(X_train2, y_train2)
ridge.score(X_train2,y_train2)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=3.58156e-22): result may not be accurate.
      overwrite_a=True).T





    0.9912533572680932




```python
ridge=Ridge(alpha=0.3)
ridge.fit(X_train2, y_train2)
ridge.score(X_train2,y_train2)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.68262e-22): result may not be accurate.
      overwrite_a=True).T





    0.991285608854472




```python
data_x.columns
```




    Index(['Auction_class', 'Bid_class', 'Claim_price', 'Appraisal_date',
           'Auction_count', 'Total_land_gross_area', 'Total_building_area',
           'Total_appraisal_price', 'First_auction_date', 'Final_auction_date',
           'addr_do', 'Apartment_usage', 'Preserve_regist_date', 'Total_floor',
           'Current_floor', 'Share_auction_YorN', 'Close_date'],
          dtype='object')



### p value 가 0.05이상인 변수 제거한 후 R-square 값 구하기


```python
data_x2=data_x.drop(['Auction_class','First_auction_date','Preserve_regist_date','Current_floor','Share_auction_YorN','Close_date'],axis=1)
```


```python
X_train3, X_test3, y_train3, y_test3 = train_test_split(data_x2, data_y, test_size=0.2, random_state=0)
```


```python
ridge=Ridge(alpha=1.0)#alpha: 얼마나 정규화를 할건지 정하는 양수 하이퍼파라미터 (클수록 더 정규화)
ridge.fit(X_train3, y_train3)
ridge.score(X_train3,y_train3)
```

    c:\users\mycom\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\linear_model\ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=2.88106e-20): result may not be accurate.
      overwrite_a=True).T





    0.9630762915948013



- feature이 줄어들었으므로, R-square값이 줄어들긴 하지만, 크게 줄어들지는 않았습니다.
