#!/bin/bash
set -e

echo "Execution started"

# # # # prepare folders
# python preparation.py
# #
# # # Execute step one
# for bz in {2..15}
# do
#   for by in {0..1}
#   do
#     for bx in {0..1}
#       do
#         python stepOne.py $bz $by $bx
#     done
#   done
# done
#
# echo "Step 1 finished"
# #
# # execute step 2 (find associatted labels_global)
# for bz in {2..15}
# do
#   python stepTwoA.py $bz
# done

python stepTwoB.py

echo "Step 2 finished"

# execute step 3 (fill holes)
for bz in {2..15}
do
  for by in {0..1}
  do
    for bx in {0..1}
    do
      python stepThree.py $bz $by $bx
    done
  done
done

echo "Step 3 finished"

#execute step 4 (optional) to verify results
python stepFour.py

echo "Step 4 finished"
