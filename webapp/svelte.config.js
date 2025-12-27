import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			pages: '../docs',
			assets: '../docs',
			fallback: 'index.html',
			precompress: false,
			strict: true,
		}),
		paths: {
			base: '/ditto-talkinghead'  
		},
		prerender: {
			entries: []
		},
		appDir: 'assets',
	}
};

export default config;
