dst=$1
echo $dst
for arg in $( cat cachedirs.txt ); do
  echo $arg
  mv $arg $dst
done

for arg in $( cat cachefiles.txt ); do
  echo $arg
  cp $arg $dst/$arg
done