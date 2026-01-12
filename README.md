> Read if you forgot the usage (even if you wrote it)
# outdated
## 1. plot level pdf for each level
1. `./sst_dump.sh -j 48 /work/tmp/s3load/fillrandom_1k_DB_280GB_*` 
generate {sst_id, key_hex, sequence} pair csv from db directory
2. `python3 merge_sst_sstable.py --csv sst_log/fillrandom_1k_DB_280GB_01_20250924_060622.csv --log=pruns_load_s3_for_seq/run_280GB_01_20250924_060622/05_sstables/stdout.log --out=merged.csv`  
merge above csv with sstables file to generate {sequence, key, level}  
sort in sequence  
3. `./merge_sst_sstables_iter.sh`  
better option with multiple db  
apply `merge_sst_sstable.py` to each db in directory   
need runs_dir, csv_dir, out_dir  

4. `python3 plot_level_sequence.py merged_scan/run_280GB_01_20250924_060622_merged.csv merged_scan/run_280GB_02_20250924_062027_merged.csv merged_scan/run_280GB_03_20250924_063441_merged.csv merged_scan/run_280GB_04_20250924_064856_merged.csv --level=1,2,3,4 --mode=pdf --out=sequence_pdf.png`

## 2. scan DB and plot hit level & TBR
-  in directory key_scan
1. `pscan/` contains raw log files to be merged
2. `python3 merge_stats.py=pscan --outdir=out -j=48` // generate merged csv log file sorted in key order
3. `python3 sample_keys.py ./ --count=4 --n=100 --out=sampled.csv` // sample 100 keys from range
4. `python3 plot_graph.py sampled.csv --metric=hit_level -o hit_level_graph.png`

## 3. scan DB and summarize diff
-  in directory key_scan
1. `python3 summarize_diff.py fillrandom_1k_280GB_001_20250828_080039_readstats.csv fillrandom_1k_280GB_002_20250828_082152_readstats.csv` // print out diff result

## 4. summarize key density among sibling dbs
1. `python3 sstable2csv.py prun_load_withseq_diffseed`  
makes sstables/*_entries.csv under root dir (in this case prun_load_with_seq_diffseed)  
2. `python3 summarize_sstables.py --sst-dir /prun_load_withseq_diffseed/sstables`  
summarize sstbles csv to one csv. default sstables_summary.csv  