// Album data configuration file
// Used to manage data for the album display page
const albumModules = import.meta.glob('../content/albums/**/*.json', { eager: true });

export interface Photo {
    src: string;
    alt?: string;
    title?: string;
    description?: string;
    tags?: string[];
    date?: string;
    width?: number;
    height?: number;
}

export interface AlbumGroup {
    id: string;
    title: string;
    description?: string;
    cover: string;
    date: string;
    location?: string;
    tags?: string[];
    layout?: "grid" | "masonry" | "list";
    columns?: number;
    photos: Photo[];
    visible?: boolean;
    basePath?: string;
}

export const albums: AlbumGroup[] = Object.entries(albumModules).map(([path, mod]: [string, any]) => {
    const id = path.split('/').pop()?.replace('.json', '') || '';
    const data = mod.default as any;
    // Get directory path relative to src/
    const basePath = path.replace('../', '').replace(/\/[^/]+$/, '');

    const album: AlbumGroup = {
        id,
        ...data,
        photos: data.photos || [],
        visible: data.visible !== false, // 默认为 true
        basePath,
    };
    return album;
});

// Sort albums by date in descending order
export const sortedAlbums = [...albums].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
);

export function groupAlbumsByFolder(albums: AlbumGroup[]): {
    folder: string;
    items: AlbumGroup[];
}[] {
    const folderMap = new Map<string, AlbumGroup[]>();
    albums.forEach(album => {
        let folder = '未分类';
        if (album.basePath) {
            // basePath 示例: 'content/albums/旅行/2024'
            const parts = album.basePath.replace(/^content\/albums\//, '').split('/');
            if (parts.length > 0 && parts[0]) {
                folder = parts[0];
            }
        }
        if (!folderMap.has(folder)) folderMap.set(folder, []);
        folderMap.get(folder)!.push(album);
    });

    const result = Array.from(folderMap.entries()).map(([folder, items]) => ({
        folder,
        items: [...items].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()),
    }));

    result.sort((a, b) => a.folder.localeCompare(b.folder));
    return result;
}