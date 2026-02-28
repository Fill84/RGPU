// SPDX-License-Identifier: GPL-2.0
/*
 * RGPU Virtual GPU Device Driver
 *
 * Creates virtual GPU device files in /dev/ that represent remote GPUs
 * managed by the RGPU daemon. The daemon communicates with this module
 * via ioctl on /dev/rgpu_control to add/remove virtual GPU devices.
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
#define RGPU_GPU_NAME_FMT   "rgpu_gpu%u"

/* Module parameters */
static int max_gpus = 16;
module_param(max_gpus, int, 0444);
MODULE_PARM_DESC(max_gpus, "Maximum number of virtual GPUs (default: 16)");

/* ---- IOCTL definitions ---- */

#define RGPU_IOCTL_MAGIC  'R'

struct rgpu_gpu_info {
	char name[128];        /* e.g., "NVIDIA GeForce RTX 3070 (Remote - RGPU)" */
	__u64 total_memory;    /* VRAM in bytes */
	__u32 index;           /* GPU index (output for ADD, input for REMOVE) */
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
	struct platform_device *pdev;
	struct miscdevice misc;
	char misc_name[32];
};

/* ---- Global state ---- */

static struct rgpu_vgpu *gpus;
static DEFINE_MUTEX(gpu_lock);
static struct miscdevice control_dev;

/* ---- GPU device file operations (minimal) ---- */

static int rgpu_gpu_open(struct inode *inode, struct file *filp)
{
	return 0;
}

static int rgpu_gpu_release(struct inode *inode, struct file *filp)
{
	return 0;
}

static ssize_t rgpu_gpu_read(struct file *filp, char __user *buf,
			     size_t count, loff_t *ppos)
{
	/* Applications can read basic info; for now return EOF */
	return 0;
}

static const struct file_operations rgpu_gpu_fops = {
	.owner   = THIS_MODULE,
	.open    = rgpu_gpu_open,
	.release = rgpu_gpu_release,
	.read    = rgpu_gpu_read,
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

	/* Create platform device for sysfs presence */
	vgpu->pdev = platform_device_alloc(RGPU_DRIVER_NAME, slot);
	if (!vgpu->pdev) {
		mutex_unlock(&gpu_lock);
		return -ENOMEM;
	}

	vgpu->pdev->dev.release = rgpu_pdev_release;
	ret = platform_device_add(vgpu->pdev);
	if (ret) {
		platform_device_put(vgpu->pdev);
		mutex_unlock(&gpu_lock);
		return ret;
	}

	/* Create /dev/rgpu_gpuN misc device */
	snprintf(vgpu->misc_name, sizeof(vgpu->misc_name),
		 RGPU_GPU_NAME_FMT, slot);

	vgpu->misc.minor = MISC_DYNAMIC_MINOR;
	vgpu->misc.name  = vgpu->misc_name;
	vgpu->misc.fops  = &rgpu_gpu_fops;
	vgpu->misc.mode  = 0666;

	ret = misc_register(&vgpu->misc);
	if (ret) {
		platform_device_unregister(vgpu->pdev);
		mutex_unlock(&gpu_lock);
		return ret;
	}

	/* Add sysfs attributes on the platform device */
	dev_set_drvdata(&vgpu->pdev->dev, vgpu);

	vgpu->active = true;

	/* Write assigned index back to userspace */
	info.index = slot;
	mutex_unlock(&gpu_lock);

	if (copy_to_user(uinfo, &info, sizeof(info)))
		return -EFAULT;

	pr_info("rgpu: added virtual GPU %u: %s (%llu MB VRAM)\n",
		slot, vgpu->name, vgpu->total_memory / (1024 * 1024));

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

	pr_info("rgpu: removing virtual GPU %u: %s\n",
		vgpu->index, vgpu->name);

	misc_deregister(&vgpu->misc);
	platform_device_unregister(vgpu->pdev);
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

	/* Register /dev/rgpu_control */
	control_dev.minor = MISC_DYNAMIC_MINOR;
	control_dev.name  = RGPU_CONTROL_NAME;
	control_dev.fops  = &rgpu_control_fops;
	control_dev.mode  = 0660;

	ret = misc_register(&control_dev);
	if (ret) {
		pr_err("rgpu: failed to register control device: %d\n", ret);
		kfree(gpus);
		return ret;
	}

	pr_info("rgpu: virtual GPU driver loaded (max_gpus=%d)\n", max_gpus);
	return 0;
}

static void __exit rgpu_vgpu_exit(void)
{
	int i;

	/* Remove all active virtual GPUs */
	mutex_lock(&gpu_lock);
	for (i = 0; i < max_gpus; i++) {
		if (gpus[i].active) {
			misc_deregister(&gpus[i].misc);
			platform_device_unregister(gpus[i].pdev);
			gpus[i].active = false;
		}
	}
	mutex_unlock(&gpu_lock);

	misc_deregister(&control_dev);
	kfree(gpus);

	pr_info("rgpu: virtual GPU driver unloaded\n");
}

module_init(rgpu_vgpu_init);
module_exit(rgpu_vgpu_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("RGPU Project");
MODULE_DESCRIPTION("RGPU Virtual GPU Device Driver");
MODULE_VERSION("0.1.0");
