// SPDX-License-Identifier: GPL-3.0-or-later
// Deadlock camera bridge for the Blender render engine (addon/gamerender.py).
//
// Deployed as server.dll in the -insecure session's mounted overlay (dlsession.py), next to the cvar unlocker
// renamed dlr_unlocker.dll: CreateInterface forwards to it. A worker thread hooks the client's main
// CViewRender::SetUpView and, while Blender sends camera packets over UDP 127.0.0.1:BRIDGE_PORT, replaces the view's
// eye, angles and FOV after the game has set it up. The player's hero is left where it is.
//
// Offsets and the camera-cache decoding come from Deadlock Dolly (cravvnn/deadlock-dolly, MIT): profile
// deadlock-2026-11-11-6861-complete, native_camera_view.hpp (layout revision 1); Copyright (c) 2026 Deadlock
// Dolly contributors, MIT licence in LICENSE-dolly.txt. Every value is checked against
// the loaded client.dll before anything is written; a different build leaves the game untouched.
//
// Freestanding (no CRT): build.sh links it with clang + lld-link against import libs made from the .def files.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;
typedef int i32;
typedef short i16;
typedef u64 uptr;
typedef void *HANDLE;
typedef int BOOL;
typedef u32 DWORD;
typedef unsigned short WCHAR;

#define WINAPI __stdcall
#define IMPORT __declspec(dllimport)
#define EXPORT __declspec(dllexport)

IMPORT HANDLE WINAPI GetModuleHandleW(const WCHAR *);
IMPORT DWORD WINAPI GetModuleFileNameW(HANDLE, WCHAR *, DWORD);
IMPORT DWORD WINAPI GetFileAttributesW(const WCHAR *);
IMPORT HANDLE WINAPI LoadLibraryExW(const WCHAR *, HANDLE, DWORD);
IMPORT void *WINAPI GetProcAddress(HANDLE, const char *);
IMPORT HANDLE WINAPI CreateThread(void *, u64, DWORD(WINAPI *)(void *), void *, DWORD, DWORD *);
IMPORT BOOL WINAPI CloseHandle(HANDLE);
IMPORT void WINAPI Sleep(DWORD);
IMPORT BOOL WINAPI VirtualProtect(void *, u64, DWORD, DWORD *);
IMPORT void *WINAPI VirtualAlloc(void *, u64, DWORD, DWORD);
IMPORT BOOL WINAPI FlushInstructionCache(HANDLE, const void *, u64);
IMPORT HANDLE WINAPI GetCurrentProcess(void);
IMPORT WCHAR *WINAPI GetCommandLineW(void);
IMPORT HANDLE WINAPI CreateFileW(const WCHAR *, DWORD, DWORD, void *, DWORD, DWORD, HANDLE);
IMPORT BOOL WINAPI WriteFile(HANDLE, const void *, DWORD, DWORD *, void *);
IMPORT BOOL WINAPI QueryPerformanceCounter(i64 *);
IMPORT BOOL WINAPI QueryPerformanceFrequency(i64 *);
IMPORT BOOL WINAPI ReadProcessMemory(HANDLE, const void *, void *, u64, u64 *);

typedef u64 SOCKET;
struct sockaddr_in {
    u16 family;
    u16 port;
    u32 addr;
    u8 zero[8];
};
IMPORT int WINAPI WSAStartup(u16, void *);
IMPORT SOCKET WINAPI socket(int, int, int);
IMPORT int WINAPI bind(SOCKET, const void *, int);
IMPORT int WINAPI recvfrom(SOCKET, char *, int, int, void *, int *);
IMPORT int WINAPI sendto(SOCKET, const char *, int, int, const void *, int);

int _fltused = 0;

void *memcpy(void *dst, const void *src, u64 n) {
    u8 *d = dst;
    const u8 *s = src;
    while (n--)
        *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, u64 n) {
    u8 *d = dst;
    while (n--)
        *d++ = (u8)c;
    return dst;
}

#define BRIDGE_PORT 29001

// Client 6861 (sha256 d0341ce7...): Dolly's reviewed profile.
#define CLIENT_IMAGE_SIZE 71380992u
#define RVA_SETUP 0x1a120a0
#define RVA_CALLER 0x1a0bac4
#define RVA_VIEW_VTABLE 0x27b8a90
#define RVA_SETUP_DELEGATE 0x5ee490
static const u8 kSetupPrologue[32] = {0x48, 0x89, 0x5c, 0x24, 0x08, 0x48, 0x89, 0x74, 0x24, 0x10, 0x57,
                                      0x48, 0x83, 0xec, 0x20, 0x48, 0x8b, 0xf1, 0xc6, 0x81, 0x30, 0x13,
                                      0x00, 0x00, 0x01, 0xb9, 0xff, 0xff, 0xff, 0xff, 0xe8, 0x3d};
// The first 15 bytes are whole instructions without RIP-relative operands: safe to relocate.
#define STOLEN 15

// CSkeletonInstance routine that builds the world-space bones for a bone mask (this = scene node, edx = mask) and
// writes the pose buffer. Found by trapping writes to a prop's pose buffer; runs on tier0 job threads after
// SetUpView, ~30 callers.
#define RVA_SKELETON 0x188b480
static const u8 kSkeletonPrologue[16] = {0x48, 0x89, 0x4c, 0x24, 0x08, 0x55, 0x53, 0x56,
                                         0x57, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41};
#define SKELETON_STOLEN 5  // mov [rsp+8], rcx

// CViewSetup fields, relative to the SetUpView object + 0x10 (layout revision 1).
#define VIEW_WIDTH 0x430
#define VIEW_HEIGHT 0x438
#define VIEW_FOV 0x498
#define VIEW_ORIGIN 0x4a0
#define VIEW_ANGLES 0x4b8
#define VIEW_ASPECT 0x4d4
#define VIEW_FLAGS 0x551
#define VIEW_AUX_ORIGIN 0x554
#define VIEW_AUX_ANGLES 0x560
#define VIEW_EXTRA_ORIGIN 0x56c

static const u8 kCachePattern[] = {
    0x4c, 0x8d, 0x0d, 0, 0, 0, 0, 0xf3, 0x0f, 0x11, 0x86, 0xd0, 0x1c, 0, 0, 0x4c, 0x8d, 0x05, 0, 0, 0, 0,
    0x48, 0x8d, 0x15, 0, 0, 0, 0, 0x48, 0x8d, 0x8e, 0xc8, 0x04, 0, 0, 0xe8, 0, 0, 0, 0, 0xf2, 0x0f, 0x10,
    0x86, 0xb0, 0x04, 0, 0, 0x48, 0x8b, 0x5c, 0x24, 0x30, 0xf2, 0x0f, 0x11, 0x05, 0, 0, 0, 0, 0x8b, 0x86,
    0xb8, 0x04, 0, 0, 0x89, 0x05, 0, 0, 0, 0, 0xf2, 0x0f, 0x10, 0x86, 0xc8, 0x04, 0, 0, 0xf2, 0x0f, 0x11,
    0x05, 0, 0, 0, 0, 0x8b, 0x86, 0xd0, 0x04, 0, 0, 0x89, 0x05, 0, 0, 0, 0};
static const u8 kCacheMasked[] = {3, 18, 25, 37, 58, 70, 86, 98};  // rel32 operands, 4 bytes each
static const u8 kAuxiliaryCopy[] = {
    0xf2, 0x0f, 0x10, 0x4c, 0x24, 0x40, 0x8b, 0x44, 0x24, 0x48, 0xf2, 0x0f, 0x11, 0x8d, 0xa0, 0x04, 0, 0,
    0xf2, 0x0f, 0x11, 0x8d, 0x54, 0x05, 0, 0, 0xf2, 0x0f, 0x11, 0x8d, 0x6c, 0x05, 0, 0, 0xf2, 0x0f, 0x10,
    0x4c, 0x24, 0x50, 0x89, 0x85, 0xa8, 0x04, 0, 0, 0x89, 0x85, 0x5c, 0x05, 0, 0, 0x89, 0x85, 0x74, 0x05,
    0, 0, 0x8b, 0x44, 0x24, 0x58, 0xf2, 0x0f, 0x11, 0x8d, 0xb8, 0x04, 0, 0, 0xf2, 0x0f, 0x11, 0x8d, 0x60,
    0x05, 0, 0, 0x89, 0x85, 0xc0, 0x04, 0, 0, 0x89, 0x85, 0x68, 0x05, 0, 0};
static const u8 kBasisEntry[] = {0x48, 0x8b, 0xc4, 0x48, 0x89, 0x58, 0x08, 0x48,
                                 0x89, 0x70, 0x10, 0x57, 0x48, 0x83, 0xec, 0x70};

typedef void(__fastcall *AngleVectors)(const float *, float *, float *, float *);
typedef void(__fastcall *SetupFn)(void *, uptr);
typedef u64(__fastcall *SkeletonFn)(uptr, u64, u64, u64);
typedef void *(*Factory)(const char *, int *);

// Global camera copies SetUpView also writes (used by culling, audio, effects).
static struct {
    uptr origin, angles, forward, right, up;
    AngleVectors basis;
} gCache;

static HANDLE gModule;
static uptr gClient;
static SetupFn gOriginal;
static SkeletonFn gSkeletonOriginal;
static Factory gUnlockerFactory;
static volatile long gStarted;
static HANDLE gLog = (HANDLE)-1;

// Latest command from Blender, published with a sequence lock (odd while writing).
struct Command {
    char magic[4];  // "DLRC"
    u32 seq;
    u32 flags;      // bit 0: override the camera
    float origin[3];
    float angles[3];  // pitch yaw roll, degrees
    float fov;        // written to the view's fov field as is (0 = keep the game's)
    float aspect;     // 0 = keep
};
#define FLAG_OVERRIDE 1u
static volatile u32 gLock;
static struct Command gCommand;
// Blender pings at least once a second; after RELEASE_AFTER seconds of silence the game gets its camera back (a
// closed or crashed Blender must not leave the game stuck on its last view).
#define RELEASE_AFTER 3
static volatile i64 gLastPacket;
static i64 gFrequency;

struct Status {
    char magic[4];  // "DLRS"
    u32 version;
    u32 state;  // 0 starting, 1 hooked, 2 refused (see message)
    u32 views;  // main views seen
    u32 applied_seq;
    i32 width, height;
    float game_origin[3];
    float game_angles[3];
    float game_fov;
    float game_aspect;
    char message[96];
};
static struct Status gStatus = {{'D', 'L', 'R', 'S'}, 1};

static void wcopy(WCHAR *dst, const WCHAR *src) {
    while ((*dst++ = *src++))
        ;
}

static u64 slen(const char *s) {
    u64 n = 0;
    while (s[n])
        n++;
    return n;
}

static void log_line(const char *text) {
    DWORD written;
    if (gLog != (HANDLE)-1) {
        WriteFile(gLog, text, (DWORD)slen(text), &written, 0);
        WriteFile(gLog, "\r\n", 2, &written, 0);
    }
}

static void set_message(const char *text) {
    u64 n = slen(text);
    if (n > sizeof(gStatus.message) - 1)
        n = sizeof(gStatus.message) - 1;
    memcpy(gStatus.message, text, n);
    gStatus.message[n] = 0;
    log_line(text);
}

static void beside_module(WCHAR *out, const WCHAR *name) {
    DWORD n = GetModuleFileNameW(gModule, out, 512);
    while (n && out[n - 1] != '\\' && out[n - 1] != '/')
        n--;
    wcopy(out + n, name);
}

// The game's own server.dll: this file sits at <game>\\citadel\\<overlay>\\bin\\win64\\server.dll.
static void game_server_dll(WCHAR *out) {
    DWORD n = GetModuleFileNameW(gModule, out, 512);
    for (int up = 0; up < 4; up++) {  // file name, win64, bin, overlay
        while (n && out[n - 1] != '\\' && out[n - 1] != '/')
            n--;
        if (up < 3 && n)
            n--;
    }
    static const WCHAR tail[] = {'b', 'i', 'n', '\\', 'w', 'i', 'n', '6', '4', '\\', 's', 'e', 'r', 'v', 'e', 'r', '.',
                                 'd', 'l', 'l', 0};
    wcopy(out + n, tail);
}

// The bridge writes into the game's memory: it only ever starts in an -insecure (no VAC) process. Matched as a
// whole argument, so "-insecure_x" or a path containing it doesn't count.
static int has_insecure(void) {
    const WCHAR *cmd = GetCommandLineW();
    static const char want[] = "-insecure";
    for (const WCHAR *p = cmd; *p; p++) {
        if (p != cmd && p[-1] != ' ' && p[-1] != '\t' && p[-1] != '"')
            continue;
        int i = 0;
        while (want[i] && p[i] == (WCHAR)want[i])
            i++;
        if (!want[i] && (p[i] == 0 || p[i] == ' ' || p[i] == '\t' || p[i] == '"'))
            return 1;
    }
    return 0;
}

static int bytes_equal(const u8 *a, const u8 *b, u64 n) {
    for (u64 i = 0; i < n; i++)
        if (a[i] != b[i])
            return 0;
    return 1;
}

static int count_pattern(const u8 *code, u64 size, const u8 *pattern, const u8 *mask, u64 length, u64 *at) {
    int found = 0;
    for (u64 i = 0; i + length <= size; i++) {
        u64 j = 0;
        while (j < length && (!mask[j] || code[i + j] == pattern[j]))
            j++;
        if (j == length) {
            if (!found)
                *at = i;
            found++;
        }
    }
    return found;
}

static int section_range(uptr module, const char *name, uptr *base, u64 *size) {
    u32 pe = *(u32 *)(module + 0x3c);
    u16 sections = *(u16 *)(module + pe + 6);
    u16 optional = *(u16 *)(module + pe + 20);
    const u8 *s = (const u8 *)(module + pe + 24 + optional);
    for (u16 i = 0; i < sections; i++, s += 40) {
        if (bytes_equal(s, (const u8 *)name, slen(name)) && !s[slen(name)]) {
            *base = module + *(u32 *)(s + 12);
            *size = *(u32 *)(s + 8);
            return 1;
        }
    }
    return 0;
}

static u32 image_size(uptr module) {
    u32 pe = *(u32 *)(module + 0x3c);
    return *(u32 *)(module + pe + 24 + 56);
}

static int within(uptr value, u64 bytes, uptr base, u64 length) {
    return value >= base && bytes <= length && value - base <= length - bytes;
}

// Dolly's decode_updated: find where SetUpView stores the global camera copies and the AngleVectors helper.
static int resolve_cache(void) {
    uptr data, text;
    u64 data_size, text_size;
    if (!section_range(gClient, ".data", &data, &data_size) || !section_range(gClient, ".text", &text, &text_size))
        return 0;
    const u8 *producer = (const u8 *)(gClient + RVA_SETUP_DELEGATE);
    u8 all[sizeof(kAuxiliaryCopy)];
    memset(all, 1, sizeof(all));
    u64 at;
    if (count_pattern(producer, 0x2ad, kAuxiliaryCopy, all, sizeof(kAuxiliaryCopy), &at) != 1)
        return 0;
    u8 mask[sizeof(kCachePattern)];
    memset(mask, 1, sizeof(mask));
    for (u64 i = 0; i < sizeof(kCacheMasked); i++)
        memset(mask + kCacheMasked[i], 0, 4);
    const u8 *code = (const u8 *)(gClient + RVA_SETUP);
    u64 offset;
    if (count_pattern(code, 0xc9, kCachePattern, mask, sizeof(kCachePattern), &offset) != 1)
        return 0;
#define TARGET(op) ((uptr)((i64)(gClient + RVA_SETUP + offset + (op) + 4) + *(const i32 *)(code + offset + (op))))
    gCache.up = TARGET(3);
    gCache.right = TARGET(18);
    gCache.forward = TARGET(25);
    gCache.origin = TARGET(58);
    gCache.angles = TARGET(86);
    uptr basis = TARGET(37);
    if (TARGET(70) != gCache.origin + 8 || TARGET(98) != gCache.angles + 8 || gCache.angles != gCache.origin + 16 ||
        gCache.forward != gCache.angles + 16 || gCache.right != gCache.forward + 16 ||
        gCache.up != gCache.right + 16 || gCache.origin % 4 || !within(gCache.origin, 76, data, data_size) ||
        !within(basis, 32, text, text_size) || !bytes_equal((const u8 *)basis, kBasisEntry, sizeof(kBasisEntry)))
        return 0;
#undef TARGET
    gCache.basis = (AngleVectors)basis;
    return 1;
}

// Entities (client 6861). The entity list global and identity layout are Dolly's reviewed FollowTarget values.
// Scene node fields match schema_detailed_class_layout; the model state was found at node+0x140 in memory (its
// model name at +0x1e8), not the schema's 0x120.
#define RVA_ENTITY_LIST 0x3633e68
#define IDENTITY_STRIDE 0x70
#define IDENTITY_HANDLE 0x10
#define ENTITY_IDENTITY 0x10
#define ENTITY_SCENE_NODE 0x330  // C_BaseEntity::m_pGameSceneNode
#define NODE_OWNER 0x30          // CGameSceneNode::m_pOwner
#define NODE_ABS_ORIGIN 0xc8     // CGameSceneNode::m_vecAbsOrigin
#define NODE_MODEL_STATE 0x140
#define STATE_BONES 0x80         // world space pose: float pos[3], scale, quat xyzw per bone (32 bytes)
#define STATE_BONE_COUNT 0x90
#define STATE_MODEL_NAME 0xa8    // CModelState::m_ModelName (char *)
#define STATE_MODEL 0xa0         // handle: points at the CModel pointer
#define MODEL_BONE_COUNT 0x160
#define MODEL_BONE_NAMES 0x168   // char *[]
#define MODEL_BONE_PARENTS 0x180 // int16[]
#define MODEL_INV_BIND 0x210     // model space inverse bind, 32-byte entries like the pose buffer
#define BIND_CHUNK 256
#define RVA_PLAYER_PAWN_VTABLE 0x27a5350  // the player's own hero is never posed
#define MAX_BONES 1024
#define POSE_SLOTS 8

static int safe_read(uptr address, void *out, u64 size) {
    u64 done = 0;
    return address > 0x10000 && ReadProcessMemory(GetCurrentProcess(), (const void *)address, out, size, &done) &&
           done == size;
}

static int contains(const char *text, const char *part) {
    for (; *text; text++) {
        u64 i = 0;
        while (part[i] && text[i] == part[i])
            i++;
        if (!part[i])
            return 1;
    }
    return 0;
}

// A slot binds one model entity to a pose sent from Blender. The net thread resolves and fills it; the hook copies
// the pose into the entity's pose buffer every frame. Slots are guarded by a seqlock like the camera command.
struct Slot {
    volatile u32 lock;
    u32 active;
    u32 entity_index;
    uptr node;
    u32 count;
    float pose[MAX_BONES][8];
};
static struct Slot gSlots[POSE_SLOTS];

static int node_model(uptr node, char *name, u64 size) {
    uptr name_ptr;
    if (!safe_read(node + NODE_MODEL_STATE + STATE_MODEL_NAME, &name_ptr, 8) || !safe_read(name_ptr, name, size - 1))
        return 0;
    name[size - 1] = 0;
    return bytes_equal((const u8 *)name, (const u8 *)"models/", 7);
}

static int claimed(uptr node, u32 except) {
    for (u32 i = 0; i < POSE_SLOTS; i++)
        if (i != except && gSlots[i].node == node && gSlots[i].active)
            return 1;
    return 0;
}

// Calls visit(context, node, index, model name) for every non-player entity with a model.
typedef int (*Visit)(void *, uptr, u32, const char *);
// The local player's pawn is left out unless asked for by index (hiding it from shots).
static void walk_entities(Visit visit, void *context, int with_pawn) {
    uptr list;
    if (!safe_read(gClient + RVA_ENTITY_LIST, &list, 8) || !list)
        return;
    for (u32 chunk_index = 0; chunk_index < 0x40; chunk_index++) {
        uptr chunk;
        if (!safe_read(list + 8 * chunk_index, &chunk, 8) || !chunk)
            continue;
        for (u32 entry = 0; entry < 512; entry++) {
            uptr identity = chunk + IDENTITY_STRIDE * entry, instance, backlink, vtable, node, owner;
            u32 handle;
            char name[128];
            if (!safe_read(identity, &instance, 8) || !instance ||
                !safe_read(instance + ENTITY_IDENTITY, &backlink, 8) || backlink != identity ||
                !safe_read(instance, &vtable, 8) || (!with_pawn && vtable == gClient + RVA_PLAYER_PAWN_VTABLE) ||
                !safe_read(instance + ENTITY_SCENE_NODE, &node, 8) || !node ||
                !safe_read(node + NODE_OWNER, &owner, 8) || owner != instance ||
                !safe_read(identity + IDENTITY_HANDLE, &handle, 4) || !node_model(node, name, sizeof(name)))
                continue;
            if (visit(context, node, handle & 0x7fff, name))
                return;
        }
    }
}

struct Find {
    const char *model;
    u32 wanted, slot, best_index;
    uptr best;
};

static int find_visit(void *context, uptr node, u32 index, const char *name) {
    struct Find *f = context;
    if (f->wanted ? index != f->wanted : (!contains(name, f->model) || claimed(node, f->slot) || index < f->best_index))
        return 0;
    f->best = node;
    f->best_index = index;
    return f->wanted != 0;
}

// Wanted entity index, or 0 for the newest unclaimed non-player entity whose model path contains `model`.
static uptr find_entity(const char *model, u32 wanted, u32 slot, u32 *index_out) {
    struct Find f = {model, wanted, slot, 0, 0};
    walk_entities(find_visit, &f, wanted != 0);
    *index_out = f.best_index;
    return f.best;
}

#define LIST_MAX 256
struct List {
    const char *model;
    u32 count;
    u32 indices[LIST_MAX];
};

static int list_visit(void *context, uptr node, u32 index, const char *name) {
    struct List *l = context;
    (void)node;
    if (contains(name, l->model) && l->count < LIST_MAX)
        l->indices[l->count++] = index;
    return 0;
}

// Runs on the job threads right after the game has built a node's world-space bones, so the pose written here is
// what the renderer draws this frame. A torn copy (Blender updating the slot meanwhile) is redone at once.
static void pose_node(uptr node) {
    for (u32 i = 0; i < POSE_SLOTS; i++) {
        struct Slot *slot = &gSlots[i];
        if (slot->node != node || !slot->active)
            continue;
        uptr bones = *(uptr *)(node + NODE_MODEL_STATE + STATE_BONES);
        u32 game_count = *(u32 *)(node + NODE_MODEL_STATE + STATE_BONE_COUNT);
        if (!bones || game_count > MAX_BONES)
            return;
        u32 before, after = 0;
        do {
            before = slot->lock;
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (before & 1)
                continue;
            u32 count = slot->count < game_count ? slot->count : game_count;
            if (!slot->active)
                return;
            memcpy((void *)bones, slot->pose, count * 32);
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            after = slot->lock;
        } while ((before & 1) || before != after);
        return;
    }
}

static u64 __fastcall skeleton_hook(uptr node, u64 mask, u64 r8, u64 r9) {
    u64 result = gSkeletonOriginal(node, mask, r8, r9);
    i64 now;
    QueryPerformanceCounter(&now);
    if (now - gLastPacket <= RELEASE_AFTER * gFrequency)
        pose_node(node);
    return result;
}

// The skeleton routine runs on job threads all the time, so it can't be patched with a multi-instruction jump
// like SetUpView. Its first instruction is 5 bytes and the entry is 16-aligned: one atomic 8-byte store puts a
// rel32 jump to a stub allocated within 2 GB of client.dll.
static int install_skeleton_hook(void) {
    u8 *target = (u8 *)(gClient + RVA_SKELETON);
    u8 *near = 0;
    for (uptr at = (gClient - 0x10000) & ~0xffffull; !near && at > gClient - 0x70000000; at -= 0x10000)
        near = VirtualAlloc((void *)at, 0x1000, 0x3000, 0x40);
    if (!near || (uptr)target % 8)
        return 0;
    u8 *tramp = near, *stub = near + 32;
    memcpy(tramp, target, SKELETON_STOLEN);
    u8 *jump = tramp + SKELETON_STOLEN;  // jmp [rip+0]; dq target+5
    jump[0] = 0xff;
    jump[1] = 0x25;
    memset(jump + 2, 0, 4);
    uptr back = (uptr)target + SKELETON_STOLEN;
    memcpy(jump + 6, &back, 8);
    stub[0] = 0xff;
    stub[1] = 0x25;
    memset(stub + 2, 0, 4);
    uptr hook = (uptr)&skeleton_hook;
    memcpy(stub + 6, &hook, 8);
    gSkeletonOriginal = (SkeletonFn)tramp;

    u64 word = *(volatile u64 *)target;
    i32 rel = (i32)((i64)stub - (i64)(target + 5));
    u8 *bytes = (u8 *)&word;
    bytes[0] = 0xe9;
    memcpy(bytes + 1, &rel, 4);
    DWORD old;
    if (!VirtualProtect(target, 8, 0x40, &old))
        return 0;
    __atomic_store_n((u64 *)target, word, __ATOMIC_SEQ_CST);
    VirtualProtect(target, 8, old, &old);
    FlushInstructionCache(GetCurrentProcess(), target, 8);
    return 1;
}

static void apply(uptr view, const float *xyz, const float *angles) {
    memcpy((void *)(view + VIEW_ORIGIN), xyz, 12);
    memcpy((void *)(view + VIEW_ANGLES), angles, 12);
    memcpy((void *)(view + VIEW_AUX_ORIGIN), xyz, 12);
    memcpy((void *)(view + VIEW_AUX_ANGLES), angles, 12);
    memcpy((void *)(view + VIEW_EXTRA_ORIGIN), xyz, 12);
    // Bit 2 makes the auxiliary view use its own pose; the main view's matrices must drive it (Dolly).
    *(u8 *)(view + VIEW_FLAGS) &= (u8)~4u;
    gCache.basis(angles, (float *)gCache.forward, (float *)gCache.right, (float *)gCache.up);
    memcpy((void *)gCache.origin, xyz, 12);
    memcpy((void *)gCache.angles, angles, 12);
}

// Game time for frame-exact renders: frozen = host_framerate at a tiny step, a step = host_framerate at the wanted
// seconds for exactly one frame (set in one SetUpView, read by the next frame, reset in the next SetUpView).
// tier0 ConVar ABI from Dolly's reviewed profile for this tier0 (sha 29eae7e1..., image 0x3fa000).
#define TIER0_IMAGE_SIZE 0x3fa000u
#define RVA_CVAR_TABLE 0x313208
#define RVA_CVAR_FIND 0x6c710
#define RVA_CVAR_DATA 0x6c650
#define CVAR_DATA_SLOT 41
#define RVA_CVAR_SET 0x20fe50
#define CVAR_TYPE 0x28   // u16: 9 float32, 10 float64 (this tier0 numbers types 2 higher than older builds)
#define CVAR_FLOAT32 9
#define CVAR_FLOAT64 10
#define CVAR_VALUE 0x58
#define FROZEN_STEP 1e-6
static const u8 kCvarSetPrologue[16] = {0x4c, 0x89, 0x44, 0x24, 0x18, 0x48, 0x89, 0x4c,
                                        0x24, 0x08, 0x55, 0x56, 0x57, 0x41, 0x54, 0x41};
struct CvarRef {
    u64 id;
    uptr data;
};
typedef void *(__fastcall *CvarFindFn)(void *, u64 *, const char *, int);
typedef uptr(__fastcall *CvarDataFn)(void *, u64);
typedef void(__fastcall *CvarSetFn)(struct CvarRef *, int, const void *, void *, void *);
static CvarSetFn gCvarSet;
static struct CvarRef gFramerate;  // host_framerate
static u16 gFramerateType;

// Requested by the net thread, carried out by the main thread in setup_hook.
static volatile u32 gTimeFrozen, gStepRequest;
static volatile float gStepSeconds;
static struct {
    char magic[4];  // "DLRT"
    u32 available, frozen, step_done;
    double framerate;
} gTime = {{'D', 'L', 'R', 'T'}};
static double gFramerateBefore;
static int gStepping;

static int resolve_cvars(void) {
    uptr tier0 = (uptr)GetModuleHandleW(L"tier0.dll"), cvar, table, find, data;
    if (!tier0 || image_size(tier0) != TIER0_IMAGE_SIZE)
        return 0;
    Factory factory = (Factory)GetProcAddress((HANDLE)tier0, "CreateInterface");
    cvar = factory ? (uptr)factory("VEngineCvar007", 0) : 0;
    if (!cvar || !safe_read(cvar, &table, 8) || table != tier0 + RVA_CVAR_TABLE ||
        !safe_read(table + 11 * 8, &find, 8) || find != tier0 + RVA_CVAR_FIND ||
        !safe_read(table + CVAR_DATA_SLOT * 8, &data, 8) || data != tier0 + RVA_CVAR_DATA ||
        !bytes_equal((const u8 *)(tier0 + RVA_CVAR_SET), kCvarSetPrologue, sizeof(kCvarSetPrologue)))
        return 0;
    gFramerate.id = 0xffffffff;
    ((CvarFindFn)find)((void *)cvar, &gFramerate.id, "host_framerate", 0);
    gFramerate.data = ((CvarDataFn)data)((void *)cvar, gFramerate.id);
    char name[16];
    uptr name_ptr;
    if (!gFramerate.data || !safe_read(gFramerate.data, &name_ptr, 8) || !safe_read(name_ptr, name, 15) ||
        !bytes_equal((const u8 *)name, (const u8 *)"host_framerate", 15) ||
        !safe_read(gFramerate.data + CVAR_TYPE, &gFramerateType, 2) || (gFramerateType != CVAR_FLOAT32 && gFramerateType != CVAR_FLOAT64))
        return 0;
    gCvarSet = (CvarSetFn)(tier0 + RVA_CVAR_SET);
    return 1;
}

static double framerate_get(void) {
    return gFramerateType == CVAR_FLOAT32 ? *(float *)(gFramerate.data + CVAR_VALUE) : *(double *)(gFramerate.data + CVAR_VALUE);
}

static void framerate_set(double value) {
    u8 typed[16] = {0};
    if (gFramerateType == CVAR_FLOAT32) {
        float f = (float)value;
        memcpy(typed, &f, 4);
    } else {
        memcpy(typed, &value, 8);
    }
    // The engine's own setter keeps change callbacks and clamping.
    gCvarSet(&gFramerate, 0, typed, (void *)(gFramerate.data + CVAR_VALUE), 0);
}

// Main thread only. A Blender that stops talking (crash, closed) gets the game clock back.
static void update_time(int talking) {
    static int tried;
    // tier0's cvar lookups take its cvar lock: only from the main thread, and not while the game is still
    // registering cvars during load (a worker-thread lookup then deadlocked the game).
    if (!tried && gStatus.views > 120) {
        tried = 1;
        gTime.available = resolve_cvars();
    }
    if (!gCvarSet)
        return;
    int frozen = gTimeFrozen && talking;
    if (frozen && !gTime.frozen) {
        gFramerateBefore = framerate_get();
        framerate_set(FROZEN_STEP);
        gTime.frozen = 1;
        gStepping = 0;
        gTime.step_done = gStepRequest;
    } else if (!frozen && gTime.frozen) {
        framerate_set(gFramerateBefore);
        gTime.frozen = 0;
        gStepping = 0;
        gTime.step_done = gStepRequest;
    } else if (frozen && gStepping) {
        framerate_set(FROZEN_STEP);
        gStepping = 0;
        gTime.step_done = gStepRequest;
    } else if (frozen && gStepRequest != gTime.step_done) {
        framerate_set(gStepSeconds);
        gStepping = 1;
    }
    if (!gTime.frozen)
        gTime.step_done = gStepRequest;
    gTime.framerate = framerate_get();
}

static void __fastcall setup_hook(void *self, uptr opaque) {
    uptr caller = (uptr)__builtin_return_address(0);
    gOriginal(self, opaque);
    if (caller != gClient + RVA_CALLER || *(uptr *)self != gClient + RVA_VIEW_VTABLE)
        return;
    uptr view = (uptr)self + 0x10;
    gStatus.views++;
    gStatus.width = *(i32 *)(view + VIEW_WIDTH);
    gStatus.height = *(i32 *)(view + VIEW_HEIGHT);
    memcpy(gStatus.game_origin, (void *)(view + VIEW_ORIGIN), 12);
    memcpy(gStatus.game_angles, (void *)(view + VIEW_ANGLES), 12);
    gStatus.game_fov = *(float *)(view + VIEW_FOV);
    gStatus.game_aspect = *(float *)(view + VIEW_ASPECT);
    i64 time_now;
    QueryPerformanceCounter(&time_now);
    update_time(time_now - gLastPacket <= RELEASE_AFTER * gFrequency);

    struct Command c;
    u32 before, after;
    do {
        before = gLock;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        c = gCommand;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        after = gLock;
    } while (before != after || (before & 1));
    i64 now;
    QueryPerformanceCounter(&now);
    if (!(c.flags & FLAG_OVERRIDE) || now - gLastPacket > RELEASE_AFTER * gFrequency)
        return;
    apply(view, c.origin, c.angles);
    if (c.fov > 1.0f && c.fov < 179.0f)
        *(float *)(view + VIEW_FOV) = c.fov;
    if (c.aspect > 0.01f)
        *(float *)(view + VIEW_ASPECT) = c.aspect;
    gStatus.applied_seq = c.seq;
}

static int install_hook(void) {
    u8 *target = (u8 *)(gClient + RVA_SETUP);
    u8 *tramp = VirtualAlloc(0, 64, 0x3000, 0x40);  // MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE
    if (!tramp)
        return 0;
    memcpy(tramp, target, STOLEN);
    u8 *jump = tramp + STOLEN;  // jmp [rip+0]; dq target+STOLEN
    jump[0] = 0xff;
    jump[1] = 0x25;
    memset(jump + 2, 0, 4);
    uptr back = (uptr)target + STOLEN;
    memcpy(jump + 6, &back, 8);
    gOriginal = (SetupFn)tramp;

    u8 patch[STOLEN];
    patch[0] = 0xff;
    patch[1] = 0x25;
    memset(patch + 2, 0, 4);
    uptr hook = (uptr)&setup_hook;
    memcpy(patch + 6, &hook, 8);
    patch[14] = 0x90;
    DWORD old;
    if (!VirtualProtect(target, STOLEN, 0x40, &old))
        return 0;
    // One 16-byte store would be atomic on x64 only if aligned; the main thread is not in SetUpView during the
    // loading screen this runs in, and Dolly's MinHook patch is no more careful.
    memcpy(target, patch, STOLEN);
    VirtualProtect(target, STOLEN, old, &old);
    FlushInstructionCache(GetCurrentProcess(), target, STOLEN);
    return 1;
}


#include "capture.c"

static DWORD WINAPI net_thread(void *arg) {
    (void)arg;
    u8 wsa[512];
    if (WSAStartup(0x0202, wsa)) {
        set_message("WSAStartup failed");
        return 0;
    }
    SOCKET s = socket(2, 2, 17);  // AF_INET, SOCK_DGRAM, IPPROTO_UDP
    struct sockaddr_in addr = {2, (u16)((BRIDGE_PORT >> 8) | ((BRIDGE_PORT & 0xff) << 8)), 0x0100007f};
    if (s == (SOCKET)~0ull || bind(s, &addr, sizeof(addr))) {
        set_message("could not bind the bridge port");
        return 0;
    }
    static u8 packet[65536];
    for (;;) {
        struct sockaddr_in from;
        int from_len = sizeof(from);
        int n = recvfrom(s, (char *)packet, sizeof(packet), 0, &from, &from_len);
        if (n < 4)
            continue;
        i64 now;
        QueryPerformanceCounter(&now);
        gLastPacket = now;
        if (n == 16 && bytes_equal(packet, (const u8 *)"DLRT", 4)) {
            // frozen u32, step request u32 (a new value = advance once), step seconds f32 -> gTime
            float seconds = *(const float *)(packet + 12);
            if (seconds > 0 && seconds < 1)
                gStepSeconds = seconds;
            gStepRequest = *(const u32 *)(packet + 8);
            gTimeFrozen = *(const u32 *)(packet + 4);
            sendto(s, (const char *)&gTime, sizeof(gTime), 0, &from, from_len);
            continue;
        }
        if (n == 264 && bytes_equal(packet, (const u8 *)"DLRF", 4)) {
            // want u32, shared file path (Windows path, ASCII)[256] -> capture status
            if (!gCap.path_set && packet[8]) {
                u32 i = 0;
                for (; i < 255 && packet[8 + i]; i++)
                    gCap.path[i] = packet[8 + i];
                gCap.path[i] = 0;
                gCap.path_set = 1;
            }
            gCap.wanted = *(const u32 *)(packet + 4);
            capture_status();
            sendto(s, (const char *)&gCapStatus, sizeof(gCapStatus), 0, &from, from_len);
            continue;
        }
        if (n == 68 && bytes_equal(packet, (const u8 *)"DLRL", 4)) {
            // model substring[64] -> "DLRL", count, entity indices (all with a model when the substring is empty)
            static struct List list;
            char model[64];
            memcpy(model, packet + 4, 64);
            model[63] = 0;
            list.model = model;
            list.count = 0;
            walk_entities(list_visit, &list, 0);
            static u8 reply[8 + 4 * LIST_MAX];
            memcpy(reply, "DLRL", 4);
            memcpy(reply + 4, &list.count, 4);
            memcpy(reply + 8, list.indices, 4 * list.count);
            sendto(s, (const char *)reply, 8 + 4 * list.count, 0, &from, from_len);
            continue;
        }
        if (n == 12 && bytes_equal(packet, (const u8 *)"DLRB", 4)) {
            // slot, first bone -> the slot's model skeleton from `first`, BIND_CHUNK bones at most per reply
            static struct {
                char magic[4];
                u32 total, first, count;
                struct {
                    float inv_bind[8];
                    i32 parent;
                    char name[60];
                } bones[BIND_CHUNK];
            } bind;
            u32 slot_index = *(const u32 *)(packet + 4), first = *(const u32 *)(packet + 8);
            uptr node = slot_index < POSE_SLOTS ? gSlots[slot_index].node : 0, handle, model = 0, names, parents, inv;
            memcpy(bind.magic, "DLRB", 4);
            bind.total = bind.count = 0;
            bind.first = first;
            if (node && safe_read(node + NODE_MODEL_STATE + STATE_MODEL, &handle, 8) && safe_read(handle, &model, 8) &&
                safe_read(model + MODEL_BONE_COUNT, &bind.total, 4) && bind.total <= MAX_BONES &&
                safe_read(model + MODEL_BONE_NAMES, &names, 8) && safe_read(model + MODEL_BONE_PARENTS, &parents, 8) &&
                safe_read(model + MODEL_INV_BIND, &inv, 8)) {
                for (u32 i = first; i < bind.total && bind.count < BIND_CHUNK; i++, bind.count++) {
                    uptr name;
                    i16 parent = -1;
                    memset(bind.bones[bind.count].name, 0, 60);
                    safe_read(inv + 32 * i, bind.bones[bind.count].inv_bind, 32);
                    safe_read(parents + 2 * i, &parent, 2);
                    bind.bones[bind.count].parent = parent;
                    if (safe_read(names + 8 * i, &name, 8))
                        safe_read(name, bind.bones[bind.count].name, 59);
                }
            } else {
                bind.total = 0;
            }
            sendto(s, (const char *)&bind, 16 + bind.count * sizeof(bind.bones[0]), 0, &from, from_len);
            continue;
        }
        if (n >= 84 && bytes_equal(packet, (const u8 *)"DLRP", 4)) {
            // slot, entity index (0 = find by model), flags (1 = pose), count, model substring[64], count x 32 bytes
            u32 slot_index = *(const u32 *)(packet + 4), wanted = *(const u32 *)(packet + 8);
            u32 flags = *(const u32 *)(packet + 12), count = *(const u32 *)(packet + 16);
            char model[64];
            memcpy(model, packet + 20, 64);
            model[63] = 0;
            struct {
                char magic[4];
                u32 found, entity_index, game_count;
                float origin[3];
                char model_name[96];
            } reply = {{'D', 'L', 'R', 'E'}};
            if (slot_index < POSE_SLOTS && count <= MAX_BONES && (u32)n == 84 + count * 32) {
                struct Slot *slot = &gSlots[slot_index];
                uptr node = slot->node;
                u32 index = slot->entity_index;
                char name[128];
                if (!node || (wanted && wanted != index) || !node_model(node, name, sizeof(name)) ||
                    !contains(name, model))
                    node = find_entity(model, wanted, slot_index, &index);
                __atomic_fetch_add(&slot->lock, 1, __ATOMIC_ACQ_REL);
                slot->node = node;
                slot->entity_index = index;
                slot->count = count;
                memcpy(slot->pose, packet + 84, count * 32);
                slot->active = node && (flags & 1) && count;
                __atomic_fetch_add(&slot->lock, 1, __ATOMIC_ACQ_REL);
                if (node && node_model(node, name, sizeof(name))) {
                    reply.found = 1;
                    reply.entity_index = index;
                    safe_read(node + NODE_MODEL_STATE + STATE_BONE_COUNT, &reply.game_count, 4);
                    safe_read(node + NODE_ABS_ORIGIN, reply.origin, 12);
                    memcpy(reply.model_name, name, sizeof(reply.model_name) - 1);
                }
            }
            sendto(s, (const char *)&reply, sizeof(reply), 0, &from, from_len);
            continue;
        }
        if (n == sizeof(struct Command) && bytes_equal(packet, (const u8 *)"DLRC", 4)) {
            __atomic_fetch_add(&gLock, 1, __ATOMIC_ACQ_REL);
            memcpy(&gCommand, packet, sizeof(struct Command));
            __atomic_fetch_add(&gLock, 1, __ATOMIC_ACQ_REL);
        }
        sendto(s, (const char *)&gStatus, sizeof(gStatus), 0, &from, from_len);
    }
}

static DWORD WINAPI worker(void *arg) {
    (void)arg;
    CloseHandle(CreateThread(0, 0, net_thread, 0, 0, 0));
    install_capture();
    HANDLE client = 0;
    for (int i = 0; i < 6000 && !client; i++) {
        client = GetModuleHandleW(L"client.dll");
        if (!client)
            Sleep(20);
    }
    if (!client) {
        gStatus.state = 2;
        set_message("client.dll never loaded");
        return 0;
    }
    gClient = (uptr)client;
    if (image_size(gClient) != CLIENT_IMAGE_SIZE ||
        !bytes_equal((const u8 *)(gClient + RVA_SETUP), kSetupPrologue, sizeof(kSetupPrologue))) {
        gStatus.state = 2;
        set_message("client.dll is not build 6861: camera hook not installed");
        return 0;
    }
    if (!resolve_cache()) {
        gStatus.state = 2;
        set_message("camera cache layout differs: camera hook not installed");
        return 0;
    }
    if (!install_hook()) {
        gStatus.state = 2;
        set_message("could not patch SetUpView");
        return 0;
    }
    gStatus.state = 1;
    if (!bytes_equal((const u8 *)(gClient + RVA_SKELETON), kSkeletonPrologue, sizeof(kSkeletonPrologue)))
        set_message("camera hook installed; skeleton routine differs: no posing");
    else if (!install_skeleton_hook())
        set_message("camera hook installed; could not patch the skeleton routine: no posing");
    else
        set_message("camera and pose hooks installed");
    return 0;
}

EXPORT void *CreateInterface(const char *name, int *result) {
    if (!gUnlockerFactory) {
        WCHAR path[600];
        // An older session layout chains through the cvar unlocker; otherwise load the game's server directly
        // (the add-on only sets cvars the release build shows).
        beside_module(path, L"dlr_unlocker.dll");
        HANDLE server = GetFileAttributesW(path) != 0xffffffffu ? LoadLibraryExW(path, 0, 0x8) : 0;
        if (!server) {
            game_server_dll(path);
            server = LoadLibraryExW(path, 0, 0x8);  // LOAD_WITH_ALTERED_SEARCH_PATH
        }
        if (server)
            gUnlockerFactory = (Factory)GetProcAddress(server, "CreateInterface");
        if (!gUnlockerFactory) {
            if (result)
                *result = 1;
            return 0;
        }
    }
    if (has_insecure() && !__atomic_exchange_n(&gStarted, 1, __ATOMIC_ACQ_REL)) {
        WCHAR path[600];
        beside_module(path, L"dlr_bridge.log");
        gLog = CreateFileW(path, 0x40000000, 1, 0, 2, 0x80, 0);  // GENERIC_WRITE, share read, CREATE_ALWAYS
        log_line("dlr bridge loaded");
        CloseHandle(CreateThread(0, 0, worker, 0, 0, 0));
    }
    return gUnlockerFactory(name, result);
}

BOOL WINAPI DllMain(HANDLE module, DWORD reason, void *reserved) {
    (void)reserved;
    if (reason == 1) {
        gModule = module;
        QueryPerformanceFrequency(&gFrequency);
    }
    return 1;
}
