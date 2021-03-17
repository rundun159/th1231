## TRIM ON
##### Cum_waf
<img src=./results/TRIM_ON/TH_TRIM_ON_cum_waf.png width="450px" height="300px" title="TRIM_OFF_cun_waf" alt="TRIM_OFF_cun_waf"></img><br/>
##### Run_waf
<img src=./results/TRIM_ON/TH_TRIM_ON_run_waf.png width="450px" height="300px" title="TRIM_OFF_run_waf" alt="TRIM_OFF_cun_waf"></img><br/>

## TRIM OFF
##### Cum_waf
<img src=./results/TRIM_OFF/TH_TRIM_OFF_cum_waf.png width="450px" height="300px" title="TRIM_OFF_cun_waf" alt="TRIM_OFF_cun_waf"></img><br/>
##### Run_waf
<img src=./results/TRIM_OFF/TH_TRIM_OFF_run_waf.png width="450px" height="300px" title="TRIM_OFF_run_waf" alt="TRIM_OFF_cun_waf"></img><br/>

- Run_waf가 40000s 이후에 감소하는것으로 그려지는 이유 : LBA의 증가가 있을때에만 run waf를 구함.   
 => 35700s, 39600s에서 run waf의 급격한 증가가 구해진 이후에 78900s까지 run waf 값이 구해지지 않다가, 그 이후에 run waf가 작은 값으로 구해져서, 선을 잇느라 지속적으로 감소하는것으로 그려짐.
 <br></br>      
- 의문점 :
  1. 78900s에서 run_waf가 작은값으로 기록되는게 맞는건가?
  2. db_size.log를 보면 용량이 갑자기 감소하는데, 제대로 실험이 진행 된건가
  3. 40000s부터 78900s까지 LBA는 바뀌지 않았다. 하나의 LBA를 채우는데 이렇게 오래걸린 정확한 이유는 무엇인가. GC를 하느라인가?
