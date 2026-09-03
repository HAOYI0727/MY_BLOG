<script lang="ts">
import { onMount } from "svelte";

import { LinkPreset, type NavbarLink } from "@/types/config";
import { LinkPresets } from "@constants/link-presets";
import { url } from "@utils/url";
import { onClickOutside } from "@utils/widget";
import { i18n } from "@i18n/translation";
import I18nKey from "@i18n/i18nKey";
import Icon from "@components/common/icon.svelte";


interface Props {
    links: NavbarLink[];
}

let { links }: Props = $props();
let isOpen = $state(false);
let expandedMenus = $state<Record<number, boolean>>({});

function resolveChildren(link: NavbarLink): NavbarLink[] {
    return (link.children || []).map((child) =>
        typeof child === "number" ? LinkPresets[child as LinkPreset] : child,
    );
}

function togglePanel() {
    isOpen = !isOpen;
    document.body.classList.toggle("mobile-nav-open", isOpen);
}

function closePanel() {
    isOpen = false;
    expandedMenus = {};
    document.body.classList.remove("mobile-nav-open");
}

function toggleSubmenu(index: number) {
    expandedMenus[index] = !expandedMenus[index];
}

// 点击外部关闭面板
function handleClickOutside(event: MouseEvent) {
    if (!isOpen) return;
    onClickOutside(event, "nav-menu-panel", "nav-menu-switch", () => {
        closePanel();
    });
}

function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape" && isOpen) {
        event.preventDefault();
        closePanel();
        document.getElementById("nav-menu-switch")?.focus();
    }
}

onMount(() => {
    document.addEventListener("click", handleClickOutside);
    document.addEventListener("keydown", handleKeydown);
    return () => {
        document.removeEventListener("click", handleClickOutside);
        document.removeEventListener("keydown", handleKeydown);
        document.body.classList.remove("mobile-nav-open");
    };
});
</script>

<div class="relative md:hidden">
    <button
        type="button"
        aria-label={isOpen ? i18n(I18nKey.closeMenu) : i18n(I18nKey.openMenu)}
        aria-expanded={isOpen}
        aria-controls="nav-menu-panel"
        name="Nav Menu"
        class="btn-plain scale-animation rounded-lg w-11 h-11 active:scale-90"
        id="nav-menu-switch"
        onclick={togglePanel}
    >
        <Icon icon={isOpen ? "material-symbols:close-rounded" : "material-symbols:menu-rounded"} class="text-[1.25rem]"></Icon>
    </button>
    {#if isOpen}
        <button
            type="button"
            class="fixed inset-0 z-40 cursor-default bg-black/35 backdrop-blur-[2px]"
            aria-label={i18n(I18nKey.closeMenu)}
            onclick={closePanel}
        ></button>
    {/if}
    <div
        id="nav-menu-panel"
        aria-hidden={!isOpen}
        inert={!isOpen}
        class="float-panel fixed z-50 transition-all right-4 left-4 sm:left-auto sm:min-w-72 px-2 py-2 max-h-[min(80dvh,42rem)] overflow-y-auto overscroll-contain"
        class:float-panel-closed={!isOpen}
    >
        {#each links as link, index}
            {@const children = resolveChildren(link)}
            {@const hasChildren = children.length > 0}
            <div class="mobile-menu-item mobile-dropdown" data-expanded={expandedMenus[index] ? "true" : "false"}>
                <div class="flex items-center rounded-lg hover:bg-(--btn-plain-bg-hover) active:bg-(--btn-plain-bg-active) transition">
                    <a href={link.external ? link.url : url(link.url)}
                        class="group flex flex-1 items-center min-w-0 py-2 pl-3 pr-2"
                        target={link.external ? "_blank" : null}
                        rel={link.external ? "noopener noreferrer" : null}
                        onclick={closePanel}
                    >
                        <span class="flex items-center transition text-black/75 dark:text-white/75 font-bold group-hover:text-(--primary) group-active:text-(--primary)">
                            {#if link.icon}
                                <Icon icon={link.icon} class="text-[1.1rem] mr-2 shrink-0" />
                            {/if}
                            {link.name}
                        </span>
                    </a>
                    {#if hasChildren}
                        <button
                            type="button"
                            class="btn-plain flex h-9 w-9 shrink-0 items-center justify-center rounded-lg mr-1"
                            aria-label={`${link.name} submenu`}
                            aria-expanded={!!expandedMenus[index]}
                            aria-controls={`mobile-submenu-${index}`}
                            onclick={() => toggleSubmenu(index)}
                        >
                            <Icon icon="material-symbols:keyboard-arrow-down-rounded" class="mobile-dropdown-arrow transition-transform text-[1.25rem] text-(--primary)" />
                        </button>
                    {:else if !link.external}
                        <Icon icon="material-symbols:chevron-right-rounded" class="mr-2 shrink-0 transition text-[1.25rem] text-(--primary)" />
                    {:else}
                        <Icon icon="fa6-solid:arrow-up-right-from-square" class="mr-2 shrink-0 transition text-[0.75rem] text-black/25 dark:text-white/25" />
                    {/if}
                </div>
                {#if hasChildren}
                    <div id={`mobile-submenu-${index}`} class="mobile-submenu" aria-hidden={!expandedMenus[index]}>
                        <div class="ml-5 mt-1 mb-1 pl-3 border-l border-black/10 dark:border-white/10">
                            {#each children as child}
                                <a
                                    href={child.external ? child.url : url(child.url)}
                                    target={child.external ? "_blank" : null}
                                    rel={child.external ? "noopener noreferrer" : null}
                                    class="group flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-75 hover:text-(--primary) hover:bg-(--btn-plain-bg-hover) transition"
                                    onclick={closePanel}
                                >
                                    {#if child.icon}
                                        <Icon icon={child.icon} class="text-[1rem] shrink-0" />
                                    {/if}
                                    <span>{child.name}</span>
                                </a>
                            {/each}
                        </div>
                    </div>
                {/if}
            </div>
        {/each}
    </div>
</div>

<style>
    :global(body.mobile-nav-open) {
        overflow: hidden;
    }
</style>
