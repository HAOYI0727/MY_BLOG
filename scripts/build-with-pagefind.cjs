/* This is a script to build the site with Pagefind */

const { execFileSync } = require('child_process');
const { existsSync } = require('fs');
const { join, resolve } = require('path');

function runLocalBinary(command, args) {
    const extension = process.platform === 'win32' ? '.cmd' : '';
    const binary = resolve('node_modules', '.bin', `${command}${extension}`);
    execFileSync(binary, args, {
        stdio: 'inherit',
        cwd: process.cwd(),
    });
}

// Detect the platform
function detectPlatform() {
    // Check environment variables
    if (process.env.GITHUB_ACTIONS) {
        return 'github';
    }
    if (process.env.CF_PAGES) {
        return 'cloudflare';
    }
    if (process.env.NETLIFY) {
        return 'netlify';
    }
    if (process.env.EDGEONE) {
        return 'edgeone';
    }
    if (process.env.VERCEL) {
        return 'vercel';
    }

    // Default to standard dist directory
    return 'default';
}

// Get Pagefind output directory
function getPagefindOutputDir(platform) {
    const outputDirs = {
        default: 'dist',
        github: 'dist',
        cloudflare: 'dist',
        netlify: 'dist',
        edgeone: 'dist',
        vercel: '.vercel/output/static',
    };

    return outputDirs[platform] || 'dist';
}

// Main function
function main() {
    const platform = detectPlatform();
    const outputDir = getPagefindOutputDir(platform);

    console.log(`🚀 Detected deployment platform: ${platform}`);
    console.log(`📁 Pagefind output directory: ${outputDir}`);

    try {
        // Run Astro build
        console.log('🔨 Running Astro build...');
        runLocalBinary('astro', ['build']);

        // Check if output directory exists
        if (!existsSync(outputDir)) {
            console.error(`❌ Output directory does not exist: ${outputDir}`);
            process.exit(1);
        }

        // Run Pagefind
        console.log(`🔍 Running Pagefind search index generation...`);
        runLocalBinary('pagefind', ['--site', outputDir]);

        console.log('🧭 Verifying reader-facing UI features...');
        execFileSync(process.execPath, [resolve('scripts', 'check-ui-features.cjs'), outputDir], {
            stdio: 'inherit',
            cwd: process.cwd(),
        });

        console.log('✅ Build completed!');
        console.log(`📊 Search index generated at: ${outputDir}/pagefind/`);

    } catch (error) {
        console.error('❌ Build failed:', error.message);
        process.exit(1);
    }
}

main();
