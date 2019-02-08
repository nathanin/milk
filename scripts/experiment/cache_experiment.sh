dst=$1
echo $dst
for arg in $( cat cachelist.txt ); do
  echo $arg
  mv $arg $dst
done
