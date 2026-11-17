// Frame capture for the D3D11 renderer (Deadlock's default on Windows, -dx11 under Proton): IDXGISwapChain::Present
// and Present1 are swapped in the swapchain class's vtable, found through a throwaway device and swapchain on a
// hidden window. In the present, before the original, the back buffer is copied into a staging texture; it is
// mapped (without waiting) a present or more later on the render thread, and the reader thread copies it into the
// shared file. Included by capture.c.
//
// No depth yet: Game Plates need the Vulkan renderer.

typedef i32 HRESULT;
typedef struct {
    u32 a;
    u16 b, c;
    u8 d[8];
} GUID;

static const GUID kIID_ID3D11Device = {0xdb6f6ddb, 0xac77, 0x4e88, {0x82, 0x53, 0x81, 0x9d, 0xf9, 0xbb, 0xf1, 0x40}};
static const GUID kIID_ID3D11Texture2D = {0x6f15aaf2, 0xd208, 0x4e89, {0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c}};
static const GUID kIID_IDXGISwapChain1 = {0x790a45f7, 0x0d42, 0x4876, {0x98, 0x3a, 0x0a, 0x55, 0xcf, 0xe6, 0xf4, 0xaa}};

// vtable slots
#define COM_RELEASE 2
#define SWAPCHAIN_GET_DEVICE 7
#define SWAPCHAIN_PRESENT 8
#define SWAPCHAIN_GET_BUFFER 9
#define SWAPCHAIN_PRESENT1 22
#define DEVICE_CREATE_TEXTURE2D 5
#define DEVICE_GET_IMMEDIATE_CONTEXT 40
#define TEXTURE2D_GET_DESC 10
#define CONTEXT_MAP 14
#define CONTEXT_UNMAP 15
#define CONTEXT_COPY_RESOURCE 47

#define DXGI_PRESENT_TEST 1u
#define D3D11_USAGE_STAGING 3u
#define D3D11_CPU_ACCESS_READ 0x20000u
#define D3D11_MAP_READ 1u
#define D3D11_MAP_FLAG_DO_NOT_WAIT 0x100000u

struct TextureDesc {
    u32 width, height, mips, array, format, samples, quality, usage, bind, cpu_access, misc;
};
struct MappedSubresource {
    void *data;
    u32 row_pitch, depth_pitch;
};
struct SwapChainDesc {
    u32 width, height, refresh_num, refresh_den, format, scanline, scaling;
    u32 samples, quality;
    u32 usage, buffer_count;
    HANDLE window;
    BOOL windowed;
    u32 swap_effect, flags;
};

typedef HRESULT(__stdcall *DxPresent)(void *, u32, u32);
typedef HRESULT(__stdcall *DxPresent1)(void *, u32, u32, const void *);
typedef HRESULT(__stdcall *DxCreateDeviceAndSwapChain)(void *, u32, HANDLE, u32, const u32 *, u32, u32,
                                                        const struct SwapChainDesc *, void **, void **, u32 *, void **);
typedef HANDLE(__stdcall *CreateWindowExWFn)(DWORD, const WCHAR *, const WCHAR *, DWORD, int, int, int, int, HANDLE,
                                             HANDLE, HANDLE, void *);
typedef BOOL(__stdcall *DestroyWindowFn)(HANDLE);

static struct {
    DxPresent present;
    DxPresent1 present1;
    void *device, *context;
    u32 width, height, dxgi_format;
    volatile u32 inside;  // the original Present may call Present1 through the same vtable
    volatile u32 failed;
} gDx;

#define VT(object, slot) ((*(void ***)(object))[slot])
#define RELEASE(object) ((u32(__stdcall *)(void *))VT(object, COM_RELEASE))(object)

// The shared file's format numbers are Vulkan's (what Blender reads).
static u32 vk_format(u32 dxgi) {
    switch (dxgi) {
    case 87:  // B8G8R8A8_UNORM
    case 90:  // B8G8R8A8_TYPELESS
        return 44;
    case 91:  // B8G8R8A8_UNORM_SRGB
        return 50;
    case 28:  // R8G8B8A8_UNORM
    case 27:  // R8G8B8A8_TYPELESS
        return 37;
    case 29:  // R8G8B8A8_UNORM_SRGB
        return 43;
    case 24:  // R10G10B10A2_UNORM
        return 64;
    case 10:  // R16G16B16A16_FLOAT
        return 97;
    }
    return 0;
}

// Ring slot states (render thread unless noted): 0 free, 1 copy issued, 2 mapped (the reader thread takes it),
// 3 reader publishing, 4 published (unmap).
static void dx_drop_staging(void) {
    for (int i = 0; i < RING; i++) {
        if (gCap.ring[i].state == 2 || gCap.ring[i].state == 4)
            ((void(__stdcall *)(void *, void *, u32))VT(gDx.context, CONTEXT_UNMAP))(gDx.context, gCap.ring[i].staging, 0);
        if (gCap.ring[i].staging)
            RELEASE(gCap.ring[i].staging);
        gCap.ring[i].staging = 0;
        gCap.ring[i].state = 0;
    }
}

static int dx_busy(void) {
    for (int i = 0; i < RING; i++)
        if (__atomic_load_n(&gCap.ring[i].state, __ATOMIC_ACQUIRE) == 3)
            return 1;
    return 0;
}

static void dx_reset(void) {
    while (dx_busy())
        Sleep(1);
    if (gDx.context) {
        dx_drop_staging();
        RELEASE(gDx.context);
    }
    if (gDx.device)
        RELEASE(gDx.device);
    gDx.context = gDx.device = 0;
    gDx.width = gDx.height = gDx.dxgi_format = 0;
}

static void dx_advance(void) {
    void *context = gDx.context;
    for (int i = 0; i < RING; i++) {
        u32 state = __atomic_load_n(&gCap.ring[i].state, __ATOMIC_ACQUIRE);
        if (state == 4) {
            ((void(__stdcall *)(void *, void *, u32))VT(context, CONTEXT_UNMAP))(context, gCap.ring[i].staging, 0);
            __atomic_store_n(&gCap.ring[i].state, 0, __ATOMIC_RELEASE);
        } else if (state == 1 && gCap.ring[i].present < gCap.presents) {
            struct MappedSubresource mapped;
            HRESULT hr = ((HRESULT(__stdcall *)(void *, void *, u32, u32, u32, struct MappedSubresource *))VT(
                context, CONTEXT_MAP))(context, gCap.ring[i].staging, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT,
                                       &mapped);
            if (hr >= 0) {
                gCap.ring[i].mapped = mapped.data;
                gCap.ring[i].pitch = mapped.row_pitch;
                __atomic_store_n(&gCap.ring[i].state, 2, __ATOMIC_RELEASE);
            }
        }
    }
}

static void dx_capture(void *swapchain) {
    void *device = 0;
    if (((HRESULT(__stdcall *)(void *, const GUID *, void **))VT(swapchain, SWAPCHAIN_GET_DEVICE))(
            swapchain, &kIID_ID3D11Device, &device) < 0 || !device) {
        cap_fail("the game's swapchain has no D3D11 device");
        return;
    }
    if (device != gDx.device) {
        dx_reset();
        gDx.device = device;
        ((void(__stdcall *)(void *, void **))VT(device, DEVICE_GET_IMMEDIATE_CONTEXT))(device, &gDx.context);
        if (!gDx.context) {
            cap_fail("no D3D11 immediate context");
            return;
        }
    } else {
        RELEASE(device);
    }
    dx_advance();
    void *back = 0;
    if (((HRESULT(__stdcall *)(void *, u32, const GUID *, void **))VT(swapchain, SWAPCHAIN_GET_BUFFER))(
            swapchain, 0, &kIID_ID3D11Texture2D, &back) < 0 || !back) {
        cap_fail("could not get the back buffer");
        return;
    }
    struct TextureDesc desc;
    ((void(__stdcall *)(void *, struct TextureDesc *))VT(back, TEXTURE2D_GET_DESC))(back, &desc);
    u32 format = vk_format(desc.format);
    if (!format || desc.samples != 1) {
        cap_fail(desc.samples != 1 ? "multisampled back buffer" : "unsupported back buffer format");
        RELEASE(back);
        return;
    }
    if (desc.width != gDx.width || desc.height != gDx.height || desc.format != gDx.dxgi_format) {
        if (dx_busy()) {
            RELEASE(back);
            return;
        }
        dx_drop_staging();
        gDx.width = desc.width;
        gDx.height = desc.height;
        gDx.dxgi_format = desc.format;
    }
    int slot = -1;
    for (int i = 0; i < RING && slot < 0; i++)
        if (__atomic_load_n(&gCap.ring[i].state, __ATOMIC_ACQUIRE) == 0)
            slot = i;
    if (slot < 0) {
        RELEASE(back);
        return;  // Blender's side is behind: skip this frame
    }
    if (!gCap.ring[slot].staging) {
        struct TextureDesc staging = desc;
        staging.mips = staging.array = 1;
        staging.quality = 0;
        staging.usage = D3D11_USAGE_STAGING;
        staging.bind = staging.misc = 0;
        staging.cpu_access = D3D11_CPU_ACCESS_READ;
        if (((HRESULT(__stdcall *)(void *, const struct TextureDesc *, const void *, void **))VT(
                gDx.device, DEVICE_CREATE_TEXTURE2D))(gDx.device, &staging, 0, &gCap.ring[slot].staging) < 0) {
            gCap.ring[slot].staging = 0;
            cap_fail("could not create the D3D11 readback texture");
            RELEASE(back);
            return;
        }
    }
    ((void(__stdcall *)(void *, void *, void *))VT(gDx.context, CONTEXT_COPY_RESOURCE))(gDx.context,
                                                                                        gCap.ring[slot].staging, back);
    RELEASE(back);
    tag_slot(slot);
    gCap.ring[slot].width = desc.width;
    gCap.ring[slot].height = desc.height;
    gCap.ring[slot].format = format;
    gCap.width = desc.width;
    gCap.height = desc.height;
    gCap.format = format;
    __atomic_store_n(&gCap.ring[slot].state, 1, __ATOMIC_RELEASE);
    if (!gCap.copied++)
        gCap.message[0] = 0;
}

static void dx_present_common(void *swapchain, u32 flags) {
    if (gDx.inside)
        return;
    count_present();
    if (!(flags & DXGI_PRESENT_TEST) && gCap.wanted && gCap.path_set)
        dx_capture(swapchain);
}

static HRESULT __stdcall dx_present_hook(void *swapchain, u32 sync, u32 flags) {
    dx_present_common(swapchain, flags);
    gDx.inside++;
    HRESULT hr = gDx.present(swapchain, sync, flags);
    gDx.inside--;
    return hr;
}

static HRESULT __stdcall dx_present1_hook(void *swapchain, u32 sync, u32 flags, const void *parameters) {
    dx_present_common(swapchain, flags);
    gDx.inside++;
    HRESULT hr = gDx.present1(swapchain, sync, flags, parameters);
    gDx.inside--;
    return hr;
}

static DWORD WINAPI dx_reader_thread(void *arg) {
    (void)arg;
    for (;;) {
        int slot = -1;
        for (int i = 0; i < RING; i++)
            if (__atomic_load_n(&gCap.ring[i].state, __ATOMIC_ACQUIRE) == 2 &&
                (slot < 0 || gCap.ring[i].present < gCap.ring[slot].present))
                slot = i;
        if (slot < 0) {
            Sleep(1);
            continue;
        }
        __atomic_store_n(&gCap.ring[slot].state, 3, __ATOMIC_RELEASE);
        publish(slot);
        __atomic_store_n(&gCap.ring[slot].state, 4, __ATOMIC_RELEASE);
    }
}

static int swap_vtable_entry(void **entry, void *hook, void **original) {
    DWORD old;
    if (!VirtualProtect(entry, 8, 0x04, &old))  // PAGE_READWRITE
        return 0;
    *original = *entry;
    __atomic_store_n(entry, hook, __ATOMIC_RELEASE);
    VirtualProtect(entry, 8, old, &old);
    return 1;
}

static void dx_fail(const char *text) {
    cap_fail(text);
    gDx.failed = 1;
}

static void install_dx11(void) {
    gCap.backend = 2;
    HANDLE d3d11 = GetModuleHandleW(L"d3d11.dll");
    HANDLE user32 = GetModuleHandleW(L"user32.dll");
    if (!d3d11)
        d3d11 = LoadLibraryExW(L"d3d11.dll", 0, 0);
    if (!user32)
        user32 = LoadLibraryExW(L"user32.dll", 0, 0);
    DxCreateDeviceAndSwapChain create =
        d3d11 ? (DxCreateDeviceAndSwapChain)GetProcAddress(d3d11, "D3D11CreateDeviceAndSwapChain") : 0;
    CreateWindowExWFn create_window = user32 ? (CreateWindowExWFn)GetProcAddress(user32, "CreateWindowExW") : 0;
    DestroyWindowFn destroy_window = user32 ? (DestroyWindowFn)GetProcAddress(user32, "DestroyWindow") : 0;
    if (!create || !create_window || !destroy_window)
        return dx_fail("D3D11: d3d11/user32 functions missing");
    // A plain STATIC control needs no window class of its own; it is never shown.
    HANDLE window = create_window(0, L"STATIC", L"dlr", 0x00CF0000, 0, 0, 64, 64, 0, 0, 0, 0);
    if (!window)
        return dx_fail("D3D11: could not create the probe window");
    void *swapchain = 0, *device = 0, *context = 0;
    u32 level;
    // Flip model like the game's swapchain; the older blit model where flip is not available.
    struct SwapChainDesc desc = {64, 64, 0, 0, 28, 0, 0, 1, 0, 0x20, 2, window, 1, 4, 0};
    HRESULT hr = create(0, 1, 0, 0, 0, 0, 7, &desc, &swapchain, &device, &level, &context);
    if (hr < 0) {
        desc.buffer_count = 1;
        desc.swap_effect = 0;
        hr = create(0, 1, 0, 0, 0, 0, 7, &desc, &swapchain, &device, &level, &context);
    }
    if (hr < 0 || !swapchain) {
        destroy_window(window);
        return dx_fail("D3D11: could not create the probe swapchain");
    }
    void **vtable = *(void ***)swapchain;
    int ok = swap_vtable_entry(&vtable[SWAPCHAIN_PRESENT], (void *)dx_present_hook, (void **)&gDx.present);
    void *swapchain1 = 0;
    if (ok && ((HRESULT(__stdcall *)(void *, const GUID *, void **))VT(swapchain, 0))(swapchain, &kIID_IDXGISwapChain1,
                                                                                      &swapchain1) >= 0 &&
        swapchain1) {
        void **vtable1 = *(void ***)swapchain1;
        swap_vtable_entry(&vtable1[SWAPCHAIN_PRESENT1], (void *)dx_present1_hook, (void **)&gDx.present1);
        RELEASE(swapchain1);
    }
    RELEASE(swapchain);
    if (context)
        RELEASE(context);
    if (device)
        RELEASE(device);
    destroy_window(window);
    if (!ok)
        return dx_fail("D3D11: could not hook Present");
    CloseHandle(CreateThread(0, 0, dx_reader_thread, 0, 0, 0));
    log_line(gDx.present1 ? "capture: D3D11 Present and Present1 hooked" : "capture: D3D11 Present hooked");
    cap_fail("D3D11: waiting for the game's present");
}
