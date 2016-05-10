#!/bin/bash

# Based on: https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7
# Install AMI: https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-763a311e


rm -rf caffe

# Get caffe, and install python requirements
git clone https://github.com/BVLC/caffe.git
cd caffe
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done

# Prepare Makefile.config so that it can build on aws
cd ../
cp Makefile.config.example Makefile.config
if [ -f "../cudnn-6.5-linux-x64-v2.tgz" ] ; then
  sed -i '/^# USE_CUDNN := 1/s/^# //' Makefile.config
fi
sed -i '/^# WITH_PYTHON_LAYER := 1/s/^# //' Makefile.config
sed -i '/^PYTHON_INCLUDE/a    /usr/local/lib/python2.7/dist-packages/numpy/core/include/ \\' Makefile.config


# edit these lines, add # to comment out
# CHECK_LE(num_axes(), 4)
#        << "Cannot use legacy accessors on Blobs with > 4 axes.";
# vi include/caffe/blob.hpp 

# Caffe takes quite a bit of disk space to build, and we don't have very much on /.
# Hence, we set the TMPDIR for to /mnt/build_tmp, under the assumption that our AMI has
# already mounted an ephemeral disk on /mnt.  Note that /mnt gets deleted on reboot, so we
# need an init script.
echo 'export TMPDIR=/mnt/build_tmp' >> Makefile.config
sudo bash -c 'cat <<EOF > /etc/init.d/create_build_dir
#!/bin/bash
if [ -d /mnt ] && [ ! -e /mnt/build_tmp ] ; then
  mkdir /mnt/build_tmp
  chown ubuntu /mnt/build_tmp
fi
EOF'
sudo chmod 744 /etc/init.d/create_build_dir
sudo /etc/init.d/create_build_dir
sudo update-rc.d create_build_dir defaults

echo 'export PYTHONPATH=/home/ubuntu/caffe/python' >> ~/.bash_profile

# And finally build!
make -j 8 all py

make -j 8 test
make runtest

# Do some cleanup
cd ../
rm -rf installation_files/