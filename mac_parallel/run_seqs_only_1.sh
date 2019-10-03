
# ./jpeg_tcm /Users/hossam.amer/7aS7aS_Works/work/my_Tools/jpeg_tcm/dataset/goose.jpg /Users/hossam.amer/7aS7aS_Works/work/my_Tools/jpeg_tcm/QF_exp/ 0

# Number of Parallel tasks (make sure that they are multiples of 11 since we are running 11 QF per image)
num_parallel_tasks=100
ID=11
last_ID=50
last_folder=4
folder=1


SECONDS=0

# for (( folder = 1; folder < 5; folder++ ))
# do
#   for (( ID_i = 1; ID_i <= 10; ID_i++ ))
#   do
while [ $folder -le $last_folder ]; do {
  while [ $ID -le $last_ID ]; do {
    # Define the input path to files
    # input_path_to_files=/Users/hossam.amer/Desktop/Projects/ML_TS/training_set_500/$folder
    # input_path_to_files=/Volumes/DATA/ml/test3
    # input_path_to_files=/Volumes/DATA/ml/test$folder
    # input_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/jpeg_ml_research/inceptionv3/inceptionv3_flowers_QF/flower_photos/roses
    # input_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/jpeg_ml_research/inceptionv3/inceptionv3_flowers_QF/flower_photos/roses
    # input_path_to_files=/Volumes/DATA/ml/ImageNet_2nd
    # input_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/my_Tools/jpeg_tcm/dataset/tcm_analysis
    # input_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/my_Tools/jpeg_tcm/dataset/set/set1
    folder=$(( (($ID-1)) /10 ))
    input_path_to_files=/Volumes/MULTICOM-104/validation_original/shard-$folder/$ID
    # input_path_to_files=/Users/ahamsala/Documents/validation_original/shard-$folder/$ID
    # input_path_to_files=/Volumes/MULTICOM-104/validation_original/shard-0/1_1

    # Define output path
    # output_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/my_Tools/jpeg_tcm/QF_exp/
    # output_path_to_files=/Users/hossam.amer/Desktop/Projects/ML_TS/train_output_folder/$folder/
    # output_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/my_Tools/jpeg_tcm/dataset/out_tcm_analysis/

    # output_path_to_files=/Volumes/DATA/ml/out_test/
    # output_path_to_files=/Users/hossam.amer/7aS7aS_Works/work/jpeg_ml_research/inceptionv3/inceptionv3_flowers_QF/flower_photos_QF/roses/
    # output_path_to_files=/Users/ahamsala/Documents/validation_generated_QF/shard-0/1/
    # output_path_to_files=/Users/ahamsala/Documents/validation_generated_QF/shard-$folder/$ID/
    output_path_to_files=/Volumes/MULTICOM-104/validation_generated_QF
    # output_path_to_files=/Users/ahamsala/Documents/validation_generated_QF
    # mkdir $output_path_to_files
    output_txt=/Volumes/MULTICOM-104/validation_generated_QF_TXT_6
    # mkdir $output_path_to_files



    # jpeg_files="$(ls $input_path_to_files)"
    # jpeg_files=(`ls -B "$input_path_to_files")
    jpeg_files=(`ls $input_path_to_files`)
    # jpeg_files=($input_path_to_files/*)
    jpeg_files_count=`ls -B "$input_path_to_files" | wc -l`

    echo 'Processing JPEG files: ' $jpeg_files_count

    # Print the jpeg files and their total number
    # echo "${jpeg_files[@]}"
    # echo $jpeg_files_count
    


    # Generate quality factors
    noQp=21
    scaleQpIndex=0
    stepSizeQp=5
    startQp=0

    # echo 'Generating Quality factors'
    # for ((i = 1; i<=noQp;i++))
    # do
    #   Qp[$i]=$(($startQp + $scaleQpIndex*$stepSizeQp))
    #   let "scaleQpIndex+=1"
    #   # echo ${Qp[$i]}
    # done


    # Generate the list of commands:


    echo 'Generating Commands list and running batches of images'

    # count the total number of commands
    commands_count=0


    # parallel group id
    group_id=0



    let "count_helper=$jpeg_files_count-1"
    for (( i = 0; i < jpeg_files_count; i++ ))
    do
      current_jpeg_help=${jpeg_files[$i]}
      current_jpeg=$input_path_to_files/$current_jpeg_help
      # echo $current_jpeg
      # exit 1
      # For every YUV, CFG, Case sequence file -> Run All Qps
      # for ((j = 1; j<=noQp;j++))
      # do
        # cmd="./jpeg_tcm $current_jpeg $output_path_to_files "${Qp[$j]}""
        # cmd="/Users/ahamsala/Library/Developer/Xcode/DerivedData/jpeg_tcm-ajysjwltnclwbibdzocnawquhenl/Build/Products/Debug/jpeg_tcm $current_jpeg $output_path_to_files 100"
        echo "$output_path_to_files"
        cmd="./JPEG_PSNR $current_jpeg $output_path_to_files $output_txt"
          
          cmd_array+=("$cmd")
          let "commands_count+=1"
          #  echo $cmd
      # done

      # echo '------------'

      # echo 'Commands Count:' $commands_count
      if [ "$(($commands_count))" == "$num_parallel_tasks" ]; then


        # Run in Parallel - Sequence level parallelism
        group_id=$(($group_id + 1))
        echo -e '\n\n New Group of id' $group_id 'will start now...\n\n'
        echo -e 'Folder id' $folder 'will start now for cmds...\n\n'
        echo $commands_count
        printf '%s\n' "${cmd_array[@]}"


        # start of the commands list is always zero
        start=0

        # end cannot exceed the length of the YUV array
        end=$(($start + $num_parallel_tasks))
        if [ "$(($end))" -gt "$num_parallel_tasks" ]; then
           end=$num_parallel_tasks
        fi


      # Time
      a=$SECONDS

      # Run in parallel
      while [  $start -lt $end ]; do {
        cmd="${cmd_array[start]}"
        echo "BLA BLA"
        echo "Process \"$start\" \"$cmd\" started";
        $cmd & pid=$!
        PID_LIST+=" $pid";
        start=$(($start + 1))
      } done

      trap "kill $PID_LIST" SIGINT
      echo "Parallel processes have started";
      wait $PID_LIST
      echo -e "\nAll processes have completed";


      # Time:
      elapsedseconds=$(( SECONDS - a ))
      echo -e "Total runtime for 200 tasks is " $elapsedseconds " seconds \n"

      # reset the commands count
      commands_count=0
      # shift your commands array to the left for the tasks you already ran
      cmd_array=("${cmd_array[@]:$num_parallel_tasks}")

      # shift your PID list (clear it)
      #PID_LIST=("${PID_LIST[@]:$num_parallel_tasks}")
      PID_LIST=("${PID_LIST[@]:1}") # group by group

      echo "PID LIST Length: " ${#PID_LIST[@]}
      echo ${PID_LIST[0]} 
    fi


    # Remaining cmds
    if [ "$(($i))" == "$count_helper" ]; then
      cmd_len=${#cmd_array[@]}
    if [ "$cmd_len" -lt "$num_parallel_tasks" ]; then

        # Run in Parallel - Sequence level parallelism
        group_id=$(($group_id + 1))
        echo -e '\n\n New Group of id' $group_id 'will start now for remaining cmds...\n\n'
        echo -e 'Folder id' $folder 'will start now for remaining cmds...\n\n'
        echo $cmd_len
        printf '%s\n' "${cmd_array[@]}"
       

        # start of the commands list is always zero
        start=0

        # end cannot exceed the length of the YUV array
        end=$(($start + $num_parallel_tasks))
        if [ "$(($end))" -gt "$cmd_len" ]; then
           end=$cmd_len
        fi

      # Time
      a=$SECONDS

      # Run in parallel
      while [  $start -lt $end ]; do {
        cmd="${cmd_array[start]}"
        echo "Process \"$start\" \"$cmd\" started";
        $cmd & pid=$!
        PID_LIST+=" $pid";
        start=$(($start + 1))
      } done

      trap "kill $PID_LIST" SIGINT
      echo "Parallel processes have started";
      wait $PID_LIST
      echo -e "\nAll processes have completed";

      # Time:
      elapsedseconds=$(( SECONDS - a ))
      echo -e "Total runtime for 100 tasks is " $elapsedseconds " seconds \n"
      

      # reset the commands count
      commands_count=0
      # shift your commands array to the left for the tasks you already ran
      cmd_array=("${cmd_array[@]:$num_parallel_tasks}")

      # shift your PID list (clear it)
      #PID_LIST=("${PID_LIST[@]:$num_parallel_tasks}")
      PID_LIST=("${PID_LIST[@]:1}") # group by group

      echo "PID LIST Length: " ${#PID_LIST[@]}
      echo ${PID_LIST[0]} 

    fi # done check inner

    fi # done check last iteration
    
    done # done for loop 



    # echo 'Total number of QP factors processed: ' $noQp
    echo 'Total number of jpeg files processed: ' $jpeg_files_count
    echo 'Total number of groups processed: ' $group_id
    # echo 'Total number of commands executed: ' $commands_count
    let "ID=ID+1"
    echo 'inner Folder:' $folder
    echo 'inner subFOlder: ' $ID

    # Clear everything
    # unset $PID_LIST
    # unset $cmd_array
  } done
  # done
  
  echo 'outer Folder:' $folder
  echo 'outer subFOlder: ' $ID
# done
  # let "folder=folder+1"
} done # folder loop
#******#******#******#******#******#******#******#******#******#******
