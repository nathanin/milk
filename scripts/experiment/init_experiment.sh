for arg in $( cat initlist.txt ); do
  echo $arg
  mkdir $arg
done
