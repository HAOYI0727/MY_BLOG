// src/utils/diary.ts
const diaryModules = import.meta.glob('../content/diary/**/*.json', { eager: true });

export interface Moment {
    id: string;
    title?: string;
    content: string;
    date: string;
    images?: string[];
    basePath?: string;
    folder?: string; // 新增：分类名称
}

export const moments: Moment[] = Object.entries(diaryModules).map(([path, mod]: [string, any]) => {
    const id = path.split('/').pop()?.replace('.json', '') || '';
    const data = mod.default as any;
    const basePath = path.replace('../', '').replace(/\/[^/]+$/, '');
    // 提取文件夹名（如 'Films'）
    const folderMatch = basePath.match(/^content\/diary\/([^/]+)/);
    const folder = folderMatch ? folderMatch[1] : undefined;
    const moment: Moment = {
        id,
        ...data,
        basePath,
        folder,
    };
    return moment;
});

export const sortedMoments = [...moments].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
);

export function groupMomentsByFolder(moments: Moment[]): {
    folder: string;
    items: Moment[];
}[] {
    const folderMap = new Map<string, Moment[]>();
    moments.forEach(moment => {
        let folder = '未分类';
        if (moment.basePath) {
            const parts = moment.basePath.replace(/^content\/diary\//, '').split('/');
            if (parts.length > 0 && parts[0]) {
                folder = parts[0];
            }
        }
        if (!folderMap.has(folder)) folderMap.set(folder, []);
        folderMap.get(folder)!.push(moment);
    });

    const result = Array.from(folderMap.entries()).map(([folder, items]) => ({
        folder,
        items: [...items].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()),
    }));

    result.sort((a, b) => a.folder.localeCompare(b.folder));
    return result;
}

export function getCoverImage(moment: Moment): string | undefined {
    return moment.images && moment.images.length > 0 ? moment.images[0] : undefined;
}