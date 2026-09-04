<script lang="ts">
import { onMount } from "svelte";

import { WALLPAPER_FULLSCREEN, WALLPAPER_BANNER, WALLPAPER_NONE } from "@constants/constants";
import {
    getEffectiveWallpaperMode,
    setWallpaperMode,
} from "@utils/wallpaper";
import { onClickOutside } from "@utils/widget";
import type { WALLPAPER_MODE } from "@/types/config";
import { siteConfig } from "@/config";
import { i18n } from "@i18n/translation";
import I18nKey from "@i18n/i18nKey";
import DropdownItem from "@/components/common/DropdownItem.svelte";
import DropdownPanel from "@/components/common/DropdownPanel.svelte";
import Icon from "@components/common/icon.svelte";


let mode: WALLPAPER_MODE = $state(siteConfig.wallpaper.mode || WALLPAPER_BANNER);
let isOpen = $state(false);

function switchWallpaperMode(newMode: WALLPAPER_MODE) {
    mode = newMode;
    setWallpaperMode(newMode);
    closePanel();
}

function openPanel() {
    isOpen = true;
}

function closePanel() {
    isOpen = false;
}

function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" && isOpen) {
        event.preventDefault();
        closePanel();
        document.getElementById("wallpaper-mode-switch")?.focus();
    }
}

// 点击外部关闭面板
function handleClickOutside(event: MouseEvent) {
    if (!isOpen) return;
    onClickOutside(event, "wallpaper-mode-panel", "wallpaper-mode-switch", () => {
        isOpen = false;
    });
}

onMount(() => {
    mode = getEffectiveWallpaperMode();
    const syncAppliedMode = (event: Event) => {
        const appliedMode = (event as CustomEvent<{ mode: WALLPAPER_MODE }>).detail?.mode;
        if (appliedMode) mode = appliedMode;
    };
    document.addEventListener("twilight:wallpaper-mode-applied", syncAppliedMode);
    document.addEventListener("click", handleClickOutside);
    document.addEventListener("keydown", handleKeydown);
    return () => {
        document.removeEventListener("click", handleClickOutside);
        document.removeEventListener("keydown", handleKeydown);
        document.removeEventListener("twilight:wallpaper-mode-applied", syncAppliedMode);
    };
});
</script>


<!-- z-50 make the panel higher than other float panels -->
<!-- svelte-ignore a11y_no_static_element_interactions (pointer leave only dismisses the already keyboard-accessible panel) -->
<div class="relative z-50" role="group" aria-label="Wallpaper mode" onmouseleave={closePanel}>
    <button
        aria-label="Wallpaper Mode"
        aria-expanded={isOpen}
        aria-controls="wallpaper-mode-panel"
        class="relative btn-plain scale-animation rounded-lg h-11 w-11 active:scale-90 flex items-center justify-center"
        id="wallpaper-mode-switch"
        onmouseenter={openPanel}
        onclick={openPanel}
    >
        <div class="absolute inset-0 flex items-center justify-center" class:opacity-0={mode !== WALLPAPER_BANNER}>
            <Icon icon="material-symbols:image-outline" class="text-[1.25rem]"></Icon>
        </div>
        <div class="absolute inset-0 flex items-center justify-center" class:opacity-0={mode !== WALLPAPER_FULLSCREEN}>
            <Icon icon="material-symbols:wallpaper" class="text-[1.25rem]"></Icon>
        </div>
        <div class="absolute inset-0 flex items-center justify-center" class:opacity-0={mode !== WALLPAPER_NONE}>
            <Icon icon="material-symbols:hide-image-outline" class="text-[1.25rem]"></Icon>
        </div>
    </button>
    <div
        id="wallpaper-mode-panel"
        aria-hidden={!isOpen}
        inert={!isOpen}
        class="absolute transition top-11 -right-2 pt-5"
        class:float-panel-closed={!isOpen}
    >
        <DropdownPanel>
            <DropdownItem
                    isActive={mode === WALLPAPER_BANNER}
                    isLast={false}
                    onclick={() => switchWallpaperMode(WALLPAPER_BANNER)}
            >
                <Icon icon="material-symbols:image-outline" class="text-[1.25rem] mr-3"></Icon>
                {i18n(I18nKey.wallpaperBanner)}
            </DropdownItem>
            <DropdownItem
                    isActive={mode === WALLPAPER_FULLSCREEN}
                    isLast={false}
                    onclick={() => switchWallpaperMode(WALLPAPER_FULLSCREEN)}
            >
                <Icon icon="material-symbols:wallpaper" class="text-[1.25rem] mr-3"></Icon>
                {i18n(I18nKey.wallpaperFullscreen)}
            </DropdownItem>
            <DropdownItem
                    isActive={mode === WALLPAPER_NONE}
                    isLast={true}
                    onclick={() => switchWallpaperMode(WALLPAPER_NONE)}
            >
                <Icon icon="material-symbols:hide-image-outline" class="text-[1.25rem] mr-3"></Icon>
                {i18n(I18nKey.wallpaperNone)}
            </DropdownItem>
        </DropdownPanel>
    </div>
</div>
