# basj script to combine vtu file, template session and basic changes
# Args:
# vtu = vtu file
# template = session file
# mintdens = lower bound for pseudocolor
# pngname = string attach before .png
# Returns:
# png and pdf image

SCRIPTDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

vtu=$1
template=$2
mintdens=$3
pngname=$4

absvtu=$(readlink -m $vtu)
ls $absvtu;
filename=$(basename $absvtu)
#echo "filename " $filename
dirout=$(dirname $absvtu)
#echo "dirname" $dir
png=${filename%.*}_${pngname}.png
#echo $png
changed=${filename%.*}.session

sed "s|vtuin|${absvtu}|" $template > ${changed}
sed -i "s|dirout|${dirout}|" ${changed}
sed -i "s|pngout|${png}|" ${changed}
sed -i "s|mintdens|${mintdens}|" ${changed}


/usr/local/visit/bin/visit -cli -nowin -s ${SCRIPTDIR}/restore_print.py ${changed}
pdf=${filename%.*}_${pngname}.pdf
pngfile="${dirout}/${png}"
pdffile="${dirout}"/${filename%.*}_${pngname}.pdf
echo $pngfile
echo $pdffile
convert $pngfile $pdffile
rm ${changed}
