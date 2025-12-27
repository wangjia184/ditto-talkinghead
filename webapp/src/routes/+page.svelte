<script lang="ts"> 
 
    import { Router, type RouteConfig } from "@mateothegreat/svelte5-router";
    import TrigFlow from "./TrigFlow.svelte";

    const routes: RouteConfig[] = [
        {
            path: "*",
            component: TrigFlow
        },
        /*
        {
            path: "products",
            component: Products
        },
        {
            path: "settings",
            component: Settings
        }
        */
    ];
</script>
 


<Router {routes} />