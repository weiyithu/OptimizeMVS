mkdir demo
cd demo
wget https://image.moeclub.org/GoogleDrive/1UlweAWvoMGTOyvSM2_vczLHtwsysEFGw
mv 1UlweAWvoMGTOyvSM2_vczLHtwsysEFGw cat1.tar.gz
tar xzf cat1.tar.gz
mv cat1/* ./
wget https://image.moeclub.org/GoogleDrive/1o8X_d9blVuerrD-dmi2n1NUaI7ZAButt
mv 1o8X_d9blVuerrD-dmi2n1NUaI7ZAButt cat13.tar.gz
tar xzf cat13.tar.gz
mv cat13/* ./
rm -r cat1
rm -r cat13
rm *.tar.gz
cd ..
