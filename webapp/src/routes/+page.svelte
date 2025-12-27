<script>
    export const prerender = false;
    export const ssr = false;
    import { Math } from "svelte-math";

    const latex = "\\frac{x}{3}";
</script>

<Math latex="y" />
<Math {latex} />
<Math {latex} displayMode />
