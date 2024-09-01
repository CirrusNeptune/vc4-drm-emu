#define _GNU_SOURCE
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <unistd.h>

#define DYLD_INTERPOSE(_replacement,_replacee) \
   __attribute__((used)) static struct{ const void* replacement; const void* replacee; } _interpose_##_replacee \
            __attribute__ ((section ("__DATA,__interpose"))) = { (const void*)(unsigned long)&_replacement, (const void*)(unsigned long)&_replacee };

#define EXPORT __attribute__((visibility("default")))


int ioctl_intercept(int* ret, int fd, unsigned long request, void* arg);

int ioctl_inject(int fd, unsigned long request, ...) {
    va_list	ap;
    void *arg;
    int ret;

    va_start(ap, request);
    arg = va_arg(ap, void *);
    va_end(ap);

    if (ioctl_intercept(&ret, fd, request, arg) == 0) {
        return ret;
    }

    return ioctl(fd, request, arg);
}

DYLD_INTERPOSE(ioctl_inject, ioctl);


int open_intercept(int* ret, const char* path, int oflag);

int open_inject(const char* path, int oflag, ...) {
	mode_t mode = 0;
    int ret;

	if (oflag & O_CREAT) {
		va_list ap;
		va_start(ap, oflag);
		/* compiler warns to pass int (not mode_t) to va_arg */
		mode = va_arg(ap, int);
		va_end(ap);
	}

    if (open_intercept(&ret, path, oflag) == 0) {
        return ret;
    }

    return open(path, oflag, mode);
}

DYLD_INTERPOSE(open_inject, open);


int close_intercept(int* ret, int fd);

int close_inject(int fd) {
    int ret;

    if (close_intercept(&ret, fd) == 0) {
        return ret;
    }

    return close(fd);
}

DYLD_INTERPOSE(close_inject, close);


int mmap_intercept(void** ret, void *addr, size_t len, int prot, int flags, int fd, off_t offset);

void *mmap_inject(void *addr, size_t len, int prot, int flags, int fd, off_t offset) {
    void* ret;

    if (mmap_intercept(&ret, addr, len, prot, flags, fd, offset) == 0) {
        return ret;
    }

    return mmap(addr, len, prot, flags, fd, offset);
}

DYLD_INTERPOSE(mmap_inject, mmap);


int munmap_inject(void *addr, size_t len) {
    munmap(addr, len);
    return 0;
}

DYLD_INTERPOSE(munmap_inject, munmap);
