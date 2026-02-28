# RGPU Virtual GPU Kernel Module Makefile
#
# Build:
#   make
#
# Install:
#   sudo make install
#   sudo depmod -a
#   sudo modprobe rgpu_vgpu
#
# DKMS (recommended for automatic rebuild on kernel updates):
#   sudo cp -r . /usr/src/rgpu-vgpu-0.1.0
#   sudo dkms add rgpu-vgpu/0.1.0
#   sudo dkms build rgpu-vgpu/0.1.0
#   sudo dkms install rgpu-vgpu/0.1.0

obj-m := rgpu_vgpu.o

KVER ?= $(shell uname -r)
KDIR ?= /lib/modules/$(KVER)/build
PWD  := $(shell pwd)

.PHONY: all modules clean install uninstall

all: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install: modules
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a $(KVER)
	install -D -m 0644 99-rgpu.rules $(DESTDIR)/etc/udev/rules.d/99-rgpu.rules
	@echo "Module installed. Load with: modprobe rgpu_vgpu"
	@echo "To load at boot, add 'rgpu_vgpu' to /etc/modules-load.d/rgpu.conf"

uninstall:
	rm -f /lib/modules/$(KVER)/extra/rgpu_vgpu.ko
	rm -f $(DESTDIR)/etc/udev/rules.d/99-rgpu.rules
	depmod -a $(KVER)
	@echo "Module uninstalled."
