// Frame capture: the game's presented images, copied on the GPU into host memory and published in a shared file
// Blender maps (/dev/shm on Linux through Wine's Z: drive). Included by bridge.c.
//
// rendersystemvulkan.dll loads vulkan-1.dll itself and keeps the device functions in a table in .data; the bridge
// swaps the vkCreateSwapchainKHR slot (adds TRANSFER_SRC usage, learns size and format) and the vkQueuePresentKHR
// slot (records the copy in front of the present). No code is patched.

IMPORT HANDLE WINAPI CreateFileMappingW(HANDLE, void *, DWORD, DWORD, DWORD, const WCHAR *);
IMPORT void *WINAPI MapViewOfFile(HANDLE, DWORD, DWORD, DWORD, u64);
IMPORT BOOL WINAPI UnmapViewOfFile(const void *);

#define VK_PRESENT_SRC 1000001002
#define VK_TRANSFER_SRC_OPTIMAL 6
#define VK_USAGE_TRANSFER_SRC 1u
#define VK_STAGE_TRANSFER 0x1000u
#define VK_STAGE_COLOR_OUTPUT 0x400u
#define VK_STAGE_BOTTOM 0x2000u
#define VK_STAGE_HOST 0x4000u
#define VK_ACCESS_COLOR_WRITE 0x100u
#define VK_ACCESS_TRANSFER_READ 0x800u
#define VK_ACCESS_TRANSFER_WRITE 0x1000u
#define VK_ACCESS_HOST_READ 0x2000u
#define VK_MEMORY_HOST_VISIBLE 2u
#define VK_MEMORY_HOST_COHERENT 4u
#define VK_MEMORY_HOST_CACHED 8u

struct VkSwapchainInfo {
    u32 type;
    const void *next;
    u32 flags;
    u64 surface;
    u32 min_images, format, color_space, width, height, layers, usage, sharing, family_count;
    const u32 *families;
    u32 transform, alpha, present_mode, clipped;
    u64 old;
};
struct VkPresentInfo {
    u32 type;
    const void *next;
    u32 wait_count;
    const u64 *waits;
    u32 swapchain_count;
    const u64 *swapchains;
    const u32 *indices;
    i32 *results;
};
struct VkSubmitInfo {
    u32 type;
    const void *next;
    u32 wait_count;
    const u64 *waits;
    const u32 *wait_stages;
    u32 command_count;
    void *const *commands;
    u32 signal_count;
    const u64 *signals;
};
struct VkImageBarrier {
    u32 type;
    const void *next;
    u32 src_access, dst_access, old_layout, new_layout, src_family, dst_family;
    u64 image;
    u32 aspect, base_mip, mips, base_layer, layers;
};
struct VkMemoryBarrier {
    u32 type;
    const void *next;
    u32 src_access, dst_access;
};
struct VkBufferImageCopy {
    u64 offset;
    u32 row_length, image_height;
    u32 aspect, mip, base_layer, layers;
    i32 x, y, z;
    u32 width, height, depth;
};
struct VkMemoryProperties {
    u32 type_count;
    struct {
        u32 flags, heap;
    } types[32];
    u32 heap_count;
    struct {
        u64 size;
        u32 flags;
    } heaps[16];
};

typedef void *(*VkGetDeviceProcAddr)(void *, const char *);
typedef i32 (*VkCreateSwapchain)(void *, const struct VkSwapchainInfo *, const void *, u64 *);
typedef i32 (*VkAcquireNextImage)(void *, u64, u64, u64, u64, u32 *);
typedef i32 (*VkQueuePresent)(void *, const struct VkPresentInfo *);
typedef i32 (*VkQueueSubmit)(void *, u32, const struct VkSubmitInfo *, u64);
typedef i32 (*VkGetSwapchainImages)(void *, u64, u32 *, u64 *);
typedef void (*VkGetDeviceQueue)(void *, u32, u32, void **);
typedef i32 (*VkCreate)(void *, const void *, const void *, u64 *);  // pool, buffer, fence, semaphore, memory
typedef i32 (*VkAllocateCommandBuffers)(void *, const void *, void **);
typedef void (*VkGetBufferRequirements)(void *, u64, u64 *);       // size, alignment, type bits
typedef i32 (*VkBindBufferMemory)(void *, u64, u64, u64);
typedef i32 (*VkMapMemory)(void *, u64, u64, u64, u32, void **);
typedef void (*VkDestroy)(void *, u64, const void *);
typedef i32 (*VkBeginCommandBuffer)(void *, const void *);
typedef i32 (*VkEndCommandBuffer)(void *);
typedef void (*VkCmdPipelineBarrier)(void *, u32, u32, u32, u32, const void *, u32, const void *, u32, const void *);
typedef void (*VkCmdCopyImageToBuffer)(void *, u64, u32, u64, u32, const struct VkBufferImageCopy *);
typedef i32 (*VkWaitForFences)(void *, u32, const u64 *, u32, u64);
typedef i32 (*VkResetFences)(void *, u32, const u64 *);
typedef i32 (*VkCreateInstance)(const void *, const void *, void **);
typedef i32 (*VkEnumeratePhysicalDevices)(void *, u32 *, void **);
typedef void (*VkGetPhysicalDeviceInfo)(void *, void *);  // properties, memory properties
typedef void (*VkDestroyInstance)(void *, const void *);

static struct {
    VkGetDeviceProcAddr get_device_proc;
    VkCreateSwapchain create_swapchain;
    VkQueuePresent present;
    VkAcquireNextImage acquire;
    VkQueueSubmit submit;
    VkGetSwapchainImages swapchain_images;
    VkGetDeviceQueue device_queue;
    VkCreate create_pool, create_buffer, create_fence, create_semaphore, allocate_memory;
    VkAllocateCommandBuffers allocate_commands;
    VkGetBufferRequirements buffer_requirements;
    VkBindBufferMemory bind_buffer;
    VkMapMemory map_memory;
    VkDestroy destroy_buffer, free_memory;
    VkBeginCommandBuffer begin;
    VkEndCommandBuffer end;
    VkCmdPipelineBarrier barrier;
    VkCmdCopyImageToBuffer copy;
    VkWaitForFences wait_fences;
    VkResetFences reset_fences;
} vk;

#define KNOWN_SWAPCHAINS 8
#define RING 3
#define MAX_IMAGES 16
#define FILE_HEADER 4096
#define FRAME_HEADER 64

// Shared file: "DLRF", version, latest frame header (0/1, ~0 = none), pad, file size; frame headers at 64 and 128.
struct FrameHeader {
    volatile u32 seq;  // odd while the bridge writes this frame
    u32 width, height, format, stride;
    u32 applied_seq, step_done, views;
    u64 present, data_offset, capacity, time;
};

static struct {
    void *device;
    struct {
        u64 handle;
        u32 width, height, format;
    } known[KNOWN_SWAPCHAINS];
    u32 known_next;
    u64 swapchain;  // the one the images below belong to
    u32 width, height, format, image_count;
    u64 images[MAX_IMAGES];
    int device_ready;  // pool, ring objects, family checked
    u32 memory_type;
    u64 pool;
    u64 buffer_size;
    struct {
        void *commands;
        u64 fence, semaphore, buffer, memory;
        u8 *mapped;
        volatile u32 state;  // 0 free, 1 submitted, 2 publishing
        u64 present;
        u32 width, height, format, applied_seq, step_done, views;
    } ring[RING];
    volatile u32 wanted;  // Blender asked for frames (and is still talking)
    WCHAR path[300];
    volatile u32 path_set;
    HANDLE file, mapping;
    u8 *view;
    u64 file_size, frame_capacity;
    volatile u64 presents, copied, published;
    u32 unknown_presents, kicks;
    volatile u32 kick;
    char message[64];
} gCap;

static struct {
    char magic[4];  // "DLRF"
    u32 state;      // 0 off, 1 capturing, 2 waiting for a swapchain made after the bridge loaded, 3 failed
    u32 width, height, format;
    u32 pad;
    u64 presents, copied, published;
    char message[64];
} gCapStatus = {{'D', 'L', 'R', 'F'}};

static void cap_fail(const char *text) {
    u64 i = 0;
    for (; text[i] && i < sizeof(gCap.message) - 1; i++)
        gCap.message[i] = text[i];
    gCap.message[i] = 0;
}

static int bytes_per_pixel(u32 format) {
    // B8G8R8A8/R8G8B8A8 UNORM/SRGB, A2R10G10B10/A2B10G10R10 UNORM: 4 bytes; R16G16B16A16_SFLOAT: 8
    if (format == 44 || format == 50 || format == 37 || format == 43 || format == 58 || format == 64)
        return 4;
    if (format == 97)
        return 8;
    return 0;
}

static i32 create_swapchain_hook(void *device, const struct VkSwapchainInfo *info, const void *alloc, u64 *out) {
    struct VkSwapchainInfo copy = *info;
    copy.usage |= VK_USAGE_TRANSFER_SRC;
    i32 result = vk.create_swapchain(device, &copy, alloc, out);
    if (result)
        return vk.create_swapchain(device, info, alloc, out);
    if (*out == gCap.swapchain)
        gCap.swapchain = 0;  // a reused handle: fetch its images again
    {
        gCap.device = device;
        u32 i = gCap.known_next++ % KNOWN_SWAPCHAINS;
        gCap.known[i].handle = *out;
        gCap.known[i].width = info->width;
        gCap.known[i].height = info->height;
        gCap.known[i].format = info->format;
    }
    return result;
}

#define LOAD(field, name) (*(void **)&vk.field = vk.get_device_proc(device, name))

static int load_device(void *device) {
    LOAD(swapchain_images, "vkGetSwapchainImagesKHR");
    LOAD(device_queue, "vkGetDeviceQueue");
    LOAD(create_pool, "vkCreateCommandPool");
    LOAD(create_buffer, "vkCreateBuffer");
    LOAD(create_fence, "vkCreateFence");
    LOAD(create_semaphore, "vkCreateSemaphore");
    LOAD(allocate_memory, "vkAllocateMemory");
    LOAD(allocate_commands, "vkAllocateCommandBuffers");
    LOAD(buffer_requirements, "vkGetBufferMemoryRequirements");
    LOAD(bind_buffer, "vkBindBufferMemory");
    LOAD(map_memory, "vkMapMemory");
    LOAD(destroy_buffer, "vkDestroyBuffer");
    LOAD(free_memory, "vkFreeMemory");
    LOAD(begin, "vkBeginCommandBuffer");
    LOAD(end, "vkEndCommandBuffer");
    LOAD(barrier, "vkCmdPipelineBarrier");
    LOAD(copy, "vkCmdCopyImageToBuffer");
    LOAD(wait_fences, "vkWaitForFences");
    LOAD(reset_fences, "vkResetFences");
    void **all = (void **)&vk.swapchain_images, **last = (void **)&vk.reset_fences;
    for (; all <= last; all++)
        if (!*all)
            return 0;
    return 1;
}

// Memory types belong to the physical device; a throwaway instance reads them (the device does not say which
// physical device it came from). The discrete GPU, or the first one.
static int pick_memory_type(u32 type_bits) {
    HANDLE loader = GetModuleHandleW(L"vulkan-1.dll");
    VkCreateInstance create = (VkCreateInstance)GetProcAddress(loader, "vkCreateInstance");
    VkEnumeratePhysicalDevices enumerate = (VkEnumeratePhysicalDevices)GetProcAddress(loader, "vkEnumeratePhysicalDevices");
    VkGetPhysicalDeviceInfo properties = (VkGetPhysicalDeviceInfo)GetProcAddress(loader, "vkGetPhysicalDeviceProperties");
    VkGetPhysicalDeviceInfo memory = (VkGetPhysicalDeviceInfo)GetProcAddress(loader, "vkGetPhysicalDeviceMemoryProperties");
    VkDestroyInstance destroy = (VkDestroyInstance)GetProcAddress(loader, "vkDestroyInstance");
    if (!create || !enumerate || !properties || !memory || !destroy)
        return -1;
    struct {
        u32 type;
        const void *next;
        u32 flags;
        const void *app;
        u32 layer_count;
        const void *layers;
        u32 extension_count;
        const void *extensions;
    } info = {1, 0, 0, 0, 0, 0, 0, 0};
    void *instance = 0, *devices[8];
    u32 count = 8;
    if (create(&info, 0, &instance) || !instance)
        return -1;
    int found = -1;
    if (enumerate(instance, &count, devices) >= 0 && count) {
        void *chosen = devices[0];
        static u8 props[2048];
        for (u32 i = 0; i < count; i++) {
            properties(devices[i], props);
            if (*(u32 *)(props + 16) == 2)  // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                chosen = devices[i];
        }
        static struct VkMemoryProperties mem;
        memory(chosen, &mem);
        u32 want[2] = {VK_MEMORY_HOST_VISIBLE | VK_MEMORY_HOST_COHERENT | VK_MEMORY_HOST_CACHED,
                       VK_MEMORY_HOST_VISIBLE | VK_MEMORY_HOST_COHERENT};
        for (int w = 0; w < 2 && found < 0; w++)
            for (u32 i = 0; i < mem.type_count; i++)
                if ((type_bits >> i & 1) && (mem.types[i].flags & want[w]) == want[w]) {
                    found = (int)i;
                    break;
                }
    }
    destroy(instance, 0);
    return found;
}

// Render thread, inside the present: pool, ring commands, fences and semaphores for the presenting device.
static int prepare_device(void *queue) {
    void *device = gCap.device, *first = 0;
    if (!load_device(device)) {
        cap_fail("device functions missing");
        return 0;
    }
    // The copy is recorded for queue family 0; only capture when that is the presenting queue.
    vk.device_queue(device, 0, 0, &first);
    if (first != queue) {
        cap_fail("the game presents from another queue family");
        return 0;
    }
    struct {
        u32 type;
        const void *next;
        u32 flags, family;
    } pool_info = {39, 0, 2, 0};  // RESET_COMMAND_BUFFER
    if (vk.create_pool(device, &pool_info, 0, &gCap.pool)) {
        cap_fail("vkCreateCommandPool failed");
        return 0;
    }
    for (int i = 0; i < RING; i++) {
        struct {
            u32 type;
            const void *next;
            u64 pool;
            u32 level, count;
        } alloc = {40, 0, gCap.pool, 0, 1};
        struct {
            u32 type;
            const void *next;
            u32 flags;
        } fence_info = {8, 0, 0}, semaphore_info = {9, 0, 0};
        if (vk.allocate_commands(device, &alloc, &gCap.ring[i].commands) ||
            vk.create_fence(device, &fence_info, 0, &gCap.ring[i].fence) ||
            vk.create_semaphore(device, &semaphore_info, 0, &gCap.ring[i].semaphore)) {
            cap_fail("could not create the copy commands");
            return 0;
        }
    }
    return 1;
}

static int ring_idle(void) {
    for (int i = 0; i < RING; i++)
        if (gCap.ring[i].state)
            return 0;
    return 1;
}

static int ensure_buffers(u64 size) {
    if (gCap.buffer_size >= size)
        return 1;
    // Only grows; the reader thread has to be done with every slot first.
    for (int tries = 0; tries < 2000 && !ring_idle(); tries++)
        Sleep(1);
    if (!ring_idle())
        return 0;
    void *device = gCap.device;
    for (int i = 0; i < RING; i++) {
        if (gCap.ring[i].buffer)
            vk.destroy_buffer(device, gCap.ring[i].buffer, 0);
        if (gCap.ring[i].memory)
            vk.free_memory(device, gCap.ring[i].memory, 0);
        gCap.ring[i].buffer = gCap.ring[i].memory = 0;
        gCap.ring[i].mapped = 0;
    }
    gCap.buffer_size = 0;
    for (int i = 0; i < RING; i++) {
        struct {
            u32 type;
            const void *next;
            u32 flags;
            u64 size;
            u32 usage, sharing, family_count;
            const u32 *families;
        } buffer_info = {12, 0, 0, size, 2, 0, 0, 0};  // TRANSFER_DST
        u64 requirements[3];
        if (vk.create_buffer(device, &buffer_info, 0, &gCap.ring[i].buffer))
            return 0;
        vk.buffer_requirements(device, gCap.ring[i].buffer, requirements);
        if (i == 0) {
            int type = pick_memory_type((u32)requirements[2]);
            if (type < 0) {
                cap_fail("no host-visible memory type");
                return 0;
            }
            gCap.memory_type = (u32)type;
        }
        struct {
            u32 type;
            const void *next;
            u64 size;
            u32 memory_type;
        } memory_info = {5, 0, requirements[0], gCap.memory_type};
        if (vk.allocate_memory(device, &memory_info, 0, &gCap.ring[i].memory) ||
            vk.bind_buffer(device, gCap.ring[i].buffer, gCap.ring[i].memory, 0) ||
            vk.map_memory(device, gCap.ring[i].memory, 0, ~0ull, 0, (void **)&gCap.ring[i].mapped)) {
            cap_fail("could not allocate the readback buffers");
            return 0;
        }
    }
    gCap.buffer_size = size;
    return 1;
}

static void record_copy(int slot, u64 image) {
    void *cmd = gCap.ring[slot].commands;
    struct {
        u32 type;
        const void *next;
        u32 flags;
        const void *inheritance;
    } begin = {42, 0, 1, 0};  // ONE_TIME_SUBMIT
    vk.begin(cmd, &begin);
    struct VkImageBarrier to_copy = {45, 0, VK_ACCESS_COLOR_WRITE, VK_ACCESS_TRANSFER_READ, VK_PRESENT_SRC,
                                     VK_TRANSFER_SRC_OPTIMAL, ~0u, ~0u, image, 1, 0, 1, 0, 1};
    vk.barrier(cmd, VK_STAGE_COLOR_OUTPUT, VK_STAGE_TRANSFER, 0, 0, 0, 0, 0, 1, &to_copy);
    struct VkBufferImageCopy region = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, gCap.width, gCap.height, 1};
    vk.copy(cmd, image, VK_TRANSFER_SRC_OPTIMAL, gCap.ring[slot].buffer, 1, &region);
    struct VkImageBarrier back = {45, 0, VK_ACCESS_TRANSFER_READ, 0, VK_TRANSFER_SRC_OPTIMAL, VK_PRESENT_SRC,
                                  ~0u, ~0u, image, 1, 0, 1, 0, 1};
    struct VkMemoryBarrier to_host = {46, 0, VK_ACCESS_TRANSFER_WRITE, VK_ACCESS_HOST_READ};
    vk.barrier(cmd, VK_STAGE_TRANSFER, VK_STAGE_BOTTOM | VK_STAGE_HOST, 0, 1, &to_host, 0, 0, 1, &back);
    vk.end(cmd);
}

static i32 acquire_hook(void *device, u64 swapchain, u64 timeout, u64 semaphore, u64 fence, u32 *index) {
    if (gCap.kick) {
        gCap.kick = 0;
        gCap.kicks++;
        return -1000001004;  // VK_ERROR_OUT_OF_DATE_KHR: nothing acquired, nothing signalled
    }
    return vk.acquire(device, swapchain, timeout, semaphore, fence, index);
}

// What a presented image shows is set up a frame or two before its present (measured: an image presented right
// after the bridge applied a camera can still show the previous one). Frames are tagged with the camera and clock
// state seen two presents earlier, so a tag never claims more than the pixels show.
static struct {
    u32 applied_seq, step_done, views;
} gTagHistory[3];

static i32 present_hook(void *queue, const struct VkPresentInfo *info) {
    gCap.presents++;
    gTagHistory[gCap.presents % 3].applied_seq = gStatus.applied_seq;
    gTagHistory[gCap.presents % 3].step_done = gTime.step_done;
    gTagHistory[gCap.presents % 3].views = gStatus.views;
    i64 now;
    QueryPerformanceCounter(&now);
    if (now - gLastPacket > RELEASE_AFTER * gFrequency)
        gCap.wanted = 0;
    if (!gCap.wanted || info->swapchain_count != 1 || info->wait_count > 8 || !gCap.path_set)
        return vk.present(queue, info);
    if (info->swapchains[0] != gCap.swapchain) {
        gCap.swapchain = 0;
        for (int i = 0; i < KNOWN_SWAPCHAINS; i++)
            if (gCap.known[i].handle == info->swapchains[0]) {
                gCap.width = gCap.known[i].width;
                gCap.height = gCap.known[i].height;
                gCap.format = gCap.known[i].format;
                gCap.swapchain = gCap.known[i].handle;
            }
        if (!gCap.swapchain) {
            // Made before the bridge's hook (no TRANSFER_SRC usage, unknown size and format). An acquire that says
            // the swapchain is out of date makes the game build a new one, as after a resize (it ignores the
            // present's result).
            gCap.unknown_presents++;
            if (gCap.unknown_presents % 60 == 1 && gCap.kicks < 4)
                gCap.kick = 1;  // the next acquire reports it
            return vk.present(queue, info);
        }
        gCap.image_count = 0;
    }
    if (!gCap.device)
        return vk.present(queue, info);
    if (gCap.device_ready == 0)
        gCap.device_ready = prepare_device(queue) ? 1 : -1;
    if (gCap.device_ready < 0)
        return vk.present(queue, info);
    if (!gCap.image_count) {
        u32 count = MAX_IMAGES;
        if (vk.swapchain_images(gCap.device, gCap.swapchain, &count, gCap.images) < 0 || !count) {
            gCap.swapchain = 0;
            return vk.present(queue, info);
        }
        gCap.image_count = count;
    }
    u32 index = info->indices[0];
    int bpp = bytes_per_pixel(gCap.format);
    if (index >= gCap.image_count || !bpp || !ensure_buffers((u64)gCap.width * gCap.height * bpp))
        return vk.present(queue, info);
    int slot = -1;
    for (int i = 0; i < RING; i++)
        if (!gCap.ring[i].state) {
            slot = i;
            break;
        }
    if (slot < 0)
        return vk.present(queue, info);  // Blender's side is behind: skip this frame
    record_copy(slot, gCap.images[index]);
    u32 stages[8];
    for (u32 i = 0; i < info->wait_count; i++)
        stages[i] = VK_STAGE_TRANSFER;
    void *cmd = gCap.ring[slot].commands;
    struct VkSubmitInfo submit = {4, 0, info->wait_count, info->waits, stages, 1, &cmd, 1, &gCap.ring[slot].semaphore};
    if (vk.submit(queue, 1, &submit, gCap.ring[slot].fence))
        return vk.present(queue, info);
    gCap.ring[slot].present = gCap.presents;
    gCap.ring[slot].width = gCap.width;
    gCap.ring[slot].height = gCap.height;
    gCap.ring[slot].format = gCap.format;
    u32 earlier = (u32)((gCap.presents + 1) % 3);  // two presents ago
    gCap.ring[slot].applied_seq = gTagHistory[earlier].applied_seq;
    gCap.ring[slot].step_done = gTagHistory[earlier].step_done;
    gCap.ring[slot].views = gTagHistory[earlier].views;
    __atomic_store_n(&gCap.ring[slot].state, 1, __ATOMIC_RELEASE);
    if (!gCap.copied++)
        gCap.message[0] = 0;  // drop "waiting for a new swapchain"
    struct VkPresentInfo waits_on_copy = *info;
    waits_on_copy.wait_count = 1;
    waits_on_copy.waits = &gCap.ring[slot].semaphore;
    return vk.present(queue, &waits_on_copy);
}

static void fast_copy(void *dst, const void *src, u64 n) {
    __asm__ volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(n) : : "memory");
}

// The shared file only grows (a shrinking file would fault Blender's mapping).
static int map_file(u64 capacity) {
    u64 size = FILE_HEADER + 2 * capacity;
    if (gCap.view && size <= gCap.file_size)
        return 1;
    if (gCap.view)
        UnmapViewOfFile(gCap.view);
    if (gCap.mapping)
        CloseHandle(gCap.mapping);
    gCap.view = 0;
    gCap.mapping = 0;
    if (!gCap.file) {
        // GENERIC_READ|WRITE, share read/write/delete, OPEN_ALWAYS
        gCap.file = CreateFileW(gCap.path, 0xC0000000, 7, 0, 4, 0x80, 0);
        if (gCap.file == (HANDLE)-1) {
            gCap.file = 0;
            cap_fail("could not open the shared frame file");
            return 0;
        }
    }
    gCap.mapping = CreateFileMappingW(gCap.file, 0, 4, (DWORD)(size >> 32), (DWORD)size, 0);  // PAGE_READWRITE
    gCap.view = gCap.mapping ? MapViewOfFile(gCap.mapping, 2, 0, 0, size) : 0;      // FILE_MAP_WRITE
    if (!gCap.view) {
        cap_fail("could not map the shared frame file");
        return 0;
    }
    gCap.file_size = size;
    gCap.frame_capacity = capacity;
    memcpy(gCap.view, "DLRF", 4);
    *(u32 *)(gCap.view + 4) = 1;
    *(volatile u32 *)(gCap.view + 8) = ~0u;
    *(u64 *)(gCap.view + 16) = size;
    for (int i = 0; i < 2; i++) {
        struct FrameHeader *h = (struct FrameHeader *)(gCap.view + 64 + FRAME_HEADER * i);
        h->data_offset = FILE_HEADER + capacity * i;
        h->capacity = capacity;
    }
    return 1;
}

static void publish(int slot) {
    u32 bpp = (u32)bytes_per_pixel(gCap.ring[slot].format);
    u64 bytes = (u64)gCap.ring[slot].width * gCap.ring[slot].height * bpp;
    if (!map_file(bytes > gCap.frame_capacity ? bytes : gCap.frame_capacity))
        return;
    u32 latest = *(volatile u32 *)(gCap.view + 8);
    u32 target = latest == 0 ? 1 : 0;
    struct FrameHeader *h = (struct FrameHeader *)(gCap.view + 64 + FRAME_HEADER * target);
    __atomic_fetch_add(&h->seq, 1, __ATOMIC_ACQ_REL);
    fast_copy(gCap.view + h->data_offset, gCap.ring[slot].mapped, bytes);
    h->width = gCap.ring[slot].width;
    h->height = gCap.ring[slot].height;
    h->format = gCap.ring[slot].format;
    h->stride = gCap.ring[slot].width * bpp;
    h->applied_seq = gCap.ring[slot].applied_seq;
    h->step_done = gCap.ring[slot].step_done;
    h->views = gCap.ring[slot].views;
    h->present = gCap.ring[slot].present;
    QueryPerformanceCounter((i64 *)&h->time);
    __atomic_fetch_add(&h->seq, 1, __ATOMIC_ACQ_REL);
    __atomic_store_n((volatile u32 *)(gCap.view + 8), target, __ATOMIC_RELEASE);
    gCap.published++;
}

// Waits for each submitted copy in present order and publishes it.
static DWORD WINAPI reader_thread(void *arg) {
    (void)arg;
    for (;;) {
        int slot = -1;
        for (int i = 0; i < RING; i++)
            if (__atomic_load_n(&gCap.ring[i].state, __ATOMIC_ACQUIRE) == 1 &&
                (slot < 0 || gCap.ring[i].present < gCap.ring[slot].present))
                slot = i;
        if (slot < 0) {
            Sleep(1);
            continue;
        }
        gCap.ring[slot].state = 2;
        if (vk.wait_fences(gCap.device, 1, &gCap.ring[slot].fence, 1, 2000000000ull) == 0)
            publish(slot);
        vk.reset_fences(gCap.device, 1, &gCap.ring[slot].fence);
        __atomic_store_n(&gCap.ring[slot].state, 0, __ATOMIC_RELEASE);
    }
}

static int swap_slot(uptr data, u64 size, uptr value, void *hook, void **original) {
    uptr *found = 0;
    for (u64 at = 0; at + 8 <= size; at += 8) {
        uptr *p = (uptr *)(data + at);
        if (*p == value) {
            if (found)
                return 0;  // not one slot: leave it alone
            found = p;
        }
    }
    if (!found)
        return 0;
    *original = (void *)value;
    __atomic_store_n(found, (uptr)hook, __ATOMIC_RELEASE);
    return 1;
}

// Called from the worker once the renderer is loaded.
static void install_capture(void) {
    HANDLE loader = GetModuleHandleW(L"vulkan-1.dll");
    uptr renderer = (uptr)GetModuleHandleW(L"rendersystemvulkan.dll"), data;
    u64 size;
    if (!loader || !renderer || !section_range(renderer, ".data", &data, &size)) {
        cap_fail("not the Vulkan renderer");
        return;
    }
    vk.get_device_proc = (VkGetDeviceProcAddr)GetProcAddress(loader, "vkGetDeviceProcAddr");
    uptr create = (uptr)GetProcAddress(loader, "vkCreateSwapchainKHR");
    uptr present = (uptr)GetProcAddress(loader, "vkQueuePresentKHR");
    uptr submit = (uptr)GetProcAddress(loader, "vkQueueSubmit");
    uptr acquire = (uptr)GetProcAddress(loader, "vkAcquireNextImageKHR");
    // Under Wine the renderer's table holds the loader's exports; on Windows it holds driver entry points (not
    // found here, so no capture there yet).
    vk.submit = (VkQueueSubmit)submit;
    if (!vk.get_device_proc || !create || !present || !submit ||
        !swap_slot(data, size, create, (void *)create_swapchain_hook, (void **)&vk.create_swapchain)) {
        cap_fail("renderer's swapchain slot not found");
        return;
    }
    if (!acquire || !swap_slot(data, size, acquire, (void *)acquire_hook, (void **)&vk.acquire)) {
        cap_fail("renderer's acquire slot not found");
        return;
    }
    if (!swap_slot(data, size, present, (void *)present_hook, (void **)&vk.present)) {
        cap_fail("renderer's present slot not found");
        return;
    }
    CloseHandle(CreateThread(0, 0, reader_thread, 0, 0, 0));
    cap_fail("waiting for a new swapchain (window resize)");
}

static void capture_status(void) {
    int known = gCap.swapchain != 0;
    gCapStatus.state = !vk.present ? 3 : gCap.device_ready < 0 ? 3 : gCap.published ? 1 : known ? 1 : 2;
    gCapStatus.width = gCap.width;
    gCapStatus.height = gCap.height;
    gCapStatus.format = gCap.format;
    gCapStatus.presents = gCap.presents;
    gCapStatus.copied = gCap.copied;
    gCapStatus.published = gCap.published;
    memcpy(gCapStatus.message, gCap.message, sizeof(gCapStatus.message));
}
