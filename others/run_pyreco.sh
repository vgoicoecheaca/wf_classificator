script='pyreco/exe/viewer.py' 
script_template='pyreco/exe/viewer_template.py' 
config='pyreco/config/viewer.ini'
config_template='pyreco/config/viewer_template.ini'

PES=4
NEVENTS=2500
MA_LIST=(20 40 60 80 100 120 160 200) 
VOV_LIST=(7 8)

HIT_FILES=( )
WF_FILES=( )

for VOV in $VOV_LIST; do
       for MA in $MA_LIST; do
		HIT_FILE=${MA}ma_${VOV}ov_hit.txt
		WF_FILE=${MA}ma_${VOV}ov_wf.txt

		HIT_FILES+=($HIT_FILE)
		WF_FILES+=($WF_FILE)
		
	        sed -e s:TOKEN_NEVENTS:$NEVENTS: \
		    -e s:TOKEN_NAME_HITS:"'$HIT_FILE'": \
		    -e s:TOKEN_NAME_WFS:"'$WF_FILE'": \
		    $script_template > $script

		sed -e s:TOKEN_MA:$MA: \
		    -e s:TOKEN_VOV:$VOV: \
		    -e s:TOKEN_PES:$PES: \
		    $config_template > $config

		python $script -o output -c $config
	done
done

python mixer.py $HIT_FILES[@] $WF_FILES[@] 20to200ma_7to8vov_5p5snr
