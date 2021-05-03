#!/bin/bash
start=$(date +%s)

# handle optional download dir
dest='../VOC/'
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ../VOC/ ..." 
    mkdir -p $dest
    cd $dest
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
		dest=$1
    echo "navigating to" $1 "..."
    cd $1
fi

# Download data
url='http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
file1='VOCtrainval_06-Nov-2007.tar'
file2='VOCtest_06-Nov-2007.tar'
file3='VOCtrainval_11-May-2012.tar'
for file in $file1 $file2 $file3; do
	echo "Downloading" $f '...'
	curl -LO $url$file
	tar -xf  $file -C $dest
	rm &f &
done

end=$(date +%s)
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
