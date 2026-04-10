// SPDX-License-Identifier: GPL-2.0
/*
 * RGPU Virtual GPU Device Driver
 *
 * Creates virtual GPU device files in /dev/ that represent remote GPUs
 * managed by the RGPU daemon. The daemon communicates with this module
 * via ioctl on /dev/rgpu_control to add/remove virtual GPU devices.
 *
 * Device nodes are created as /dev/nvidia{N} using NVIDIA's major number
 * (195) with the minor number supplied by userspace. This makes them
 * discoverable by nvidia-container-toolkit without patching the toolkit.
 *
 * Copyright (C) 2026 RGPU Project
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/ioctl.h>
#include <linux/cdev.h>
#include <linux/device.h>

#define RGPU_DRIVER_NAME    "rgpu_vgpu"
#define RGPU_CONTROL_NAME   "rgpu_control"

/* Module parameters */
static int max_gpus = 16;
module_param(max_gpus, int, 0444);
MODULE_PARM_DESC(max_gpus, "Maximum number of virtual GPUs (default: 16)");

/* NVIDIA uses major number 195 for all /dev/nvidia* devices */
#define NVIDIA_MAJOR 195

/* Device class — we try "nvidia" first; fall back if a real driver claimed it */
static struct class *nvidia_class;

/* ---- IOCTL definitions ---- */

#define RGPU_IOCTL_MAGIC  'R'

struct rgpu_gpu_info {
	char name[128];        /* e.g., "NVIDIA GeForce RTX 3070 (Remote - RGPU)" */
	__u64 total_memory;    /* VRAM in bytes */
	__u32 index;           /* GPU index (output for ADD, input for REMOVE) */
	__u32 minor_number;    /* nvidia minor number to create (e.g. 1 -> /dev/nvidia1) */
};

struct rgpu_gpu_list {
	__u32 count;           /* number of active GPUs */
	__u32 max_count;       /* size of the infos array (input) */
	struct rgpu_gpu_info infos[]; /* flexible array of GPU info structs */
};

#define RGPU_IOCTL_ADD_GPU    _IOWR(RGPU_IOCTL_MAGIC, 1, struct rgpu_gpu_info)
#define RGPU_IOCTL_REMOVE_GPU _IOW(RGPU_IOCTL_MAGIC, 2, struct rgpu_gpu_info)
#define RGPU_IOCTL_LIST_GPUS  _IOWR(RGPU_IOCTL_MAGIC, 3, struct rgpu_gpu_list)

/* ---- Per-GPU state ---- */

struct rgpu_vgpu {
	bool active;
	char name[128];
	u64 total_memory;
	u32 index;
	u32 nvidia_minor;         /* the /dev/nvidia{N} minor we created */
	struct platform_device *pdev;
	struct cdev cdev;         /* character device (replaces miscdevice) */
	struct device *dev;       /* sysfs device node */
	dev_t devno;              /* major:minor combo */
};

/* ---- Global state ---- */

static struct rgpu_vgpu *gpus;
static DEFINE_MUTEX(gpu_lock);
static struct miscdevice control_dev;

/* ---- GPU device file operations (minimal) ---- */
/*
 * These handlers are intentionally minimal. The device nodes only need to
 * EXIST and be openable. Real GPU ioctls are intercepted at the library
 * level (librgpu_cuda_interpose.so / librgpu_vk_icd.so) before any device
 * I/O reaches this driver.
 */

static int rgpu_gpu_open(struct inode *inode, struct file *filp)
{
	return 0;
}

static int rgpu_gpu_release(struct inode *inode, struct file *filp)
{
	return 0;
}

static long rgpu_gpu_ioctl(struct file *filp, unsigned int cmd,
			   unsigned long arg)
{
	/*
	 * Return -ENODEV for all ioctls. Our interpose libraries handle
	 * everything at the library level; nothing should reach here in
	 * normal operation.
	 */
	return -ENODEV;
}

static const struct file_operations rgpu_gpu_fops = {
	.owner          = THIS_MODULE,
	.open           = rgpu_gpu_open,
	.release        = rgpu_gpu_release,
	.unlocked_ioctl = rgpu_gpu_ioctl,
	.compat_ioctl   = rgpu_gpu_ioctl,
};

/* ---- Platform device release ---- */

static void rgpu_pdev_release(struct device *dev)
{
	/* Nothing to free; lifetime managed by rgpu_vgpu array */
}

/* ---- Add a virtual GPU ---- */

static int rgpu_add_gpu(struct rgpu_gpu_info __user *uinfo)
{
	struct rgpu_gpu_info info;
	struct rgpu_vgpu *vgpu;
	dev_t devno;
	int i, ret, slot = -1;

	if (copy_from_user(&info, uinfo, sizeof(info)))
		return -EFAULT;

	info.name[sizeof(info.name) - 1] = '\0';

	mutex_lock(&gpu_lock);

	/* Find a free slot */
	for (i = 0; i < max_gpus; i++) {
		if (!gpus[i].active) {
			slot = i;
			break;
		}
	}

	if (slot < 0) {
		mutex_unlock(&gpu_lock);
		return -ENOSPC;
	}

	vgpu = &gpus[slot];
	memset(vgpu, 0, sizeof(*vgpu));
	strscpy(vgpu->name, info.name, sizeof(vgpu->name));
	vgpu->total_memory = info.total_memory;
	vgpu->index = slot;
	vgpu->nvidia_minor = info.minor_number;

	/* Build the dev_t for /dev/nvidia{minor} */
	devno = MKDEV(NVIDIA_MAJOR, info.minor_number);
	vgpu->devno = devno;

	/* Register the character device with the kernel */
	cdev_init(&vgpu->cdev, &rgpu_gpu_fops);
	vgpu->cdev.owner = THIS_MODULE;

	ret = cdev_add(&vgpu->cdev, devno, 1);
	if (ret) {
		pr_err("rgpu: cdev_add failed for nvidia%u: %d\n",
		       info.minor_number, ret);
		mutex_unlock(&gpu_lock);
		return ret;
	}

	/* Create the /dev/nvidia{N} node via sysfs / udev */
	vgpu->dev = device_create(nvidia_class, NULL, devno, NULL,
				  "nvidia%u", info.minor_number);
	if (IS_ERR(vgpu->dev)) {
		ret = PTR_ERR(vgpu->dev);
		pr_err("rgpu: device_create failed for nvidia%u: %d\n",
		       info.minor_number, ret);
		cdev_del(&vgpu->cdev);
		mutex_unlock(&gpu_lock);
		return ret;
	}

	/* Create platform device for sysfs presence */
	vgpu->pdev = platform_device_alloc(RGPU_DRIVER_NAME, slot);
	if (!vgpu->pdev) {
		device_destroy(nvidia_class, devno);
		cdev_del(&vgpu->cdev);
		mutex_unlock(&gpu_lock);
		return -ENOMEM;
	}

	vgpu->pdev->dev.release = rgpu_pdev_release;
	ret = platform_device_add(vgpu->pdev);
	if (ret) {
		platform_device_put(vgpu->pdev);
		device_destroy(nvidia_class, devno);
		cdev_del(&vgpu->cdev);
		mutex_unlock(&gpu_lock);
		return ret;
	}

	dev_set_drvdata(&vgpu->pdev->dev, vgpu);
	vgpu->active = true;

	/* Write assigned index back to userspace */
	info.index = slot;
	mutex_unlock(&gpu_lock);

	if (copy_to_user(uinfo, &info, sizeof(info)))
		return -EFAULT;

	pr_info("rgpu: added /dev/nvidia%u -> slot %u: %s (%llu MB VRAM)\n",
		vgpu->nvidia_minor, slot, vgpu->name,
		vgpu->total_memory / (1024 * 1024));

	return 0;
}

/* ---- Remove a virtual GPU ---- */

static int rgpu_remove_gpu(struct rgpu_gpu_info __user *uinfo)
{
	struct rgpu_gpu_info info;
	struct rgpu_vgpu *vgpu;

	if (copy_from_user(&info, uinfo, sizeof(info)))
		return -EFAULT;

	if (info.index >= (u32)max_gpus)
		return -EINVAL;

	mutex_lock(&gpu_lock);

	vgpu = &gpus[info.index];
	if (!vgpu->active) {
		mutex_unlock(&gpu_lock);
		return -ENODEV;
	}

	pr_info("rgpu: removing /dev/nvidia%u (slot %u): %s\n",
		vgpu->nvidia_minor, vgpu->index, vgpu->name);

	platform_device_unregister(vgpu->pdev);
	device_destroy(nvidia_class, vgpu->devno);
	cdev_del(&vgpu->cdev);
	vgpu->active = false;

	mutex_unlock(&gpu_lock);
	return 0;
}

/* ---- List virtual GPUs ---- */

static int rgpu_list_gpus(struct rgpu_gpu_list __user *ulist)
{
	u32 max_count, count = 0;
	int i;

	if (get_user(max_count, &ulist->max_count))
		return -EFAULT;

	mutex_lock(&gpu_lock);

	for (i = 0; i < max_gpus && count < max_count; i++) {
		if (!gpus[i].active)
			continue;

		if (copy_to_user(&ulist->infos[count].name,
				 gpus[i].name, sizeof(gpus[i].name))) {
			mutex_unlock(&gpu_lock);
			return -EFAULT;
		}
		if (put_user(gpus[i].total_memory,
			     &ulist->infos[count].total_memory)) {
			mutex_unlock(&gpu_lock);
			return -EFAULT;
		}
		if (put_user(gpus[i].index,
			     &ulist->infos[count].index)) {
			mutex_unlock(&gpu_lock);
			return -EFAULT;
		}
		if (put_user(gpus[i].nvidia_minor,
			     &ulist->infos[count].minor_number)) {
			mutex_unlock(&gpu_lock);
			return -EFAULT;
		}
		count++;
	}

	mutex_unlock(&gpu_lock);

	if (put_user(count, &ulist->count))
		return -EFAULT;

	return 0;
}

/* ---- Control device ioctl ---- */

static long rgpu_control_ioctl(struct file *filp, unsigned int cmd,
			       unsigned long arg)
{
	switch (cmd) {
	case RGPU_IOCTL_ADD_GPU:
		return rgpu_add_gpu((struct rgpu_gpu_info __user *)arg);
	case RGPU_IOCTL_REMOVE_GPU:
		return rgpu_remove_gpu((struct rgpu_gpu_info __user *)arg);
	case RGPU_IOCTL_LIST_GPUS:
		return rgpu_list_gpus((struct rgpu_gpu_list __user *)arg);
	default:
		return -ENOTTY;
	}
}

static const struct file_operations rgpu_control_fops = {
	.owner          = THIS_MODULE,
	.unlocked_ioctl = rgpu_control_ioctl,
	.compat_ioctl   = rgpu_control_ioctl,
};

/* ---- Module init/exit ---- */

static int __init rgpu_vgpu_init(void)
{
	int ret;

	if (max_gpus < 1 || max_gpus > 256) {
		pr_err("rgpu: max_gpus must be between 1 and 256\n");
		return -EINVAL;
	}

	gpus = kcalloc(max_gpus, sizeof(*gpus), GFP_KERNEL);
	if (!gpus)
		return -ENOMEM;

	/*
	 * Create the device class used for device_create(). We try "nvidia"
	 * first so that udev sees the same subsystem name as the real driver.
	 * If a real nvidia driver is already loaded and owns that class name,
	 * we fall back to "rgpu_nvidia" to avoid a collision.
	 */
	nvidia_class = class_create("nvidia");
	if (IS_ERR(nvidia_class)) {
		pr_warn("rgpu: class 'nvidia' already exists, using 'rgpu_nvidia'\n");
		nvidia_class = class_create("rgpu_nvidia");
		if (IS_ERR(nvidia_class)) {
			ret = PTR_ERR(nvidia_class);
			pr_err("rgpu: failed to create device class: %d\n", ret);
			kfree(gpus);
			return ret;
		}
	}

	/* Register /dev/rgpu_control */
	control_dev.minor = MISC_DYNAMIC_MINOR;
	control_dev.name  = RGPU_CONTROL_NAME;
	control_dev.fops  = &rgpu_control_fops;
	control_dev.mode  = 0660;

	ret = misc_register(&control_dev);
	if (ret) {
		pr_err("rgpu: failed to register control device: %d\n", ret);
		class_destroy(nvidia_class);
		kfree(gpus);
		return ret;
	}

	pr_info("rgpu: virtual GPU driver loaded (max_gpus=%d, major=%d)\n",
		max_gpus, NVIDIA_MAJOR);
	return 0;
}

static void __exit rgpu_vgpu_exit(void)
{
	int i;

	/* Remove all active virtual GPUs */
	mutex_lock(&gpu_lock);
	for (i = 0; i < max_gpus; i++) {
		if (gpus[i].active) {
			platform_device_unregister(gpus[i].pdev);
			device_destroy(nvidia_class, gpus[i].devno);
			cdev_del(&gpus[i].cdev);
			gpus[i].active = false;
		}
	}
	mutex_unlock(&gpu_lock);

	misc_deregister(&control_dev);
	class_destroy(nvidia_class);
	kfree(gpus);

	pr_info("rgpu: virtual GPU driver unloaded\n");
}

module_init(rgpu_vgpu_init);
module_exit(rgpu_vgpu_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("RGPU Project");
MODULE_DESCRIPTION("RGPU Virtual GPU Device Driver — creates /dev/nvidia{N}");
MODULE_VERSION("0.2.0");
