# Salidas Perf

```sh
#!/bin/bash

nohup ./call5.sh > call5.log 2>&1 &
echo $! > pid.txt

#!/bin/bash

nohup ./call5.sh > call5.log 2>&1 &
echo $! > pid.txt
[root@nodo2 mhSearch]# cat benchmark/call5.sh
#! /bin/bash

echo "horaEjecucion,tiempo(segundos),dcache_load_misses,dcache_loads,L1_dcache_stores,LLC_loads,e_core,e_ram,cycles,instructions,cache_misses,all_per_cache,cache_references,branches" > salidamon5.csv

while true; 
    do ./mon5.sh ; 
    sleep 0.25;
done

```

```python
for dev in devlist:
    print("holis")
```
