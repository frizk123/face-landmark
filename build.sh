sudo rm bin/face_landmark
cd build
sudo make -j8
cd ..
bin/face_landmark
echo "Yes!"
