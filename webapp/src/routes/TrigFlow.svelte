<script lang="ts">
    import { Math } from "svelte-math";
</script>

<div class="container mt-4 mb-4">
    <h1>TrigFlow <a href="https://arxiv.org/html/2410.11081v2" target="_blank">arxiv.org/html/2410.11081v2</a></h1>
    <hr class="my-4" />

    <section class="mb-4">
        <h2 class="mt-4 mb-3">Basic Setup</h2>
        <p>
            TrigFlow is a theoretical framework that unifies EDM (Elucidating the Design Space of Diffusion-Based Generative Models), 
            Flow Matching, and Velocity Prediction through a simplified parameterization. The key insight is to use trigonometric 
            functions to parameterize the diffusion process, which significantly simplifies the formulation of diffusion models, 
            the associated probability flow ODE, and consistency models. This unification eliminates the need for complex noise 
            schedules and provides a more elegant mathematical foundation for training generative models.
        </p>
    </section>

    <section class="mb-4">
        <h2 class="mt-4 mb-3">Diffusion Process</h2>
        <p>
            In TrigFlow, the diffusion process is parameterized using trigonometric functions. Given a data sample <Math displayMode={false}>x</Math> 
            from the target distribution and noise <Math displayMode={false}>{"z \\sim \\mathcal{N}(0, I)"}</Math> from a standard normal distribution, 
            the noisy sample at time <Math displayMode={false}>t</Math> is defined as:
        </p>
        <p class="fs-3">
            <Math displayMode={true}>{"x_t = \\cos(t) \\cdot x + \\sin(t) \\cdot z"}</Math>
        </p>
        <p>
            where <Math displayMode={false}>{"t \\in [0, \\pi/2]"}</Math>. This parameterization ensures that at <Math displayMode={false}>t = 0</Math>, 
            we have <Math displayMode={false}>x_0 = x</Math> (pure data), and at <Math displayMode={false}>{"t = \\pi/2"}</Math>, 
            we have <Math displayMode={false}>{"x_{\\pi/2} = z"}</Math> (pure noise).
        </p>
    </section>

    <section class="mb-4">
        <h2 class="mt-4 mb-3">PF-ODE</h2>
        <p>
            The probability flow ODE (PF-ODE) describes the evolution of the diffusion process. The velocity is defined as the time derivative of the position:
            <Math displayMode={false}>{"v_t = \\frac{dx_t}{dt}"}</Math>
        </p>
       
        <p>
            Taking the derivative of <Math displayMode={false}>x_t = \cos(t) \cdot x + \sin(t) \cdot z</Math> with respect to time <Math displayMode={false}>t</Math>, we obtain:
        </p>
        <p class="fs-3">
            <Math displayMode={true}>{"v_t = \\frac{dx_t}{dt} = -\\sin(t) \\cdot x + \\cos(t) \\cdot z"}</Math>
        </p> 
    </section>


    <section class="mb-4">
        <h2 class="mt-4 mb-3">Deriving the Consistency Property</h2>
        <p>
            Starting from the diffusion process and velocity definitions:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"x_t = \\cos(t) \\cdot x_0 + \\sin(t) \\cdot z"}</Math><br />
            <Math displayMode={true}>{"v_t = -\\sin(t) \\cdot x_0 + \\cos(t) \\cdot z"}</Math>
        </p>
        <p>
            Solving this linear system for <Math displayMode={false}>x_0</Math> and <Math displayMode={false}>z</Math>. 
            We can write it in matrix form:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"\\begin{bmatrix} x_t \\\\ v_t \\end{bmatrix} = \\begin{bmatrix} \\cos(t) & \\sin(t) \\\\ -\\sin(t) & \\cos(t) \\end{bmatrix} \\begin{bmatrix} x_0 \\\\ z \\end{bmatrix}"}</Math>
        </p>
        <p>
            The coefficient matrix is a rotation matrix with determinant <Math displayMode={false}>{"\\cos^2(t) + \\sin^2(t) = 1"}</Math>, so its inverse is its transpose:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"\\begin{bmatrix} x_0 \\\\ z \\end{bmatrix} = \\begin{bmatrix} \\cos(t) & -\\sin(t) \\\\ \\sin(t) & \\cos(t) \\end{bmatrix} \\begin{bmatrix} x_t \\\\ v_t \\end{bmatrix}"}</Math>
        </p>
        <p>
            Expanding this, we obtain:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"x_0 = \\cos(t) \\cdot x_t - \\sin(t) \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"z = \\sin(t) \\cdot x_t + \\cos(t) \\cdot v_t"}</Math>
        </p>
        <p>
            Now, consider a state at a different time <Math displayMode={false}>r</Math>, where <Math displayMode={false}>{"r \\in [0, \\pi/2]"}</Math> and <Math displayMode={false}>r</Math> can be any time point (not necessarily related to <Math displayMode={false}>t</Math>). 
            By the same diffusion process definition:
        </p>
        <p class="fs-4">
            <Math displayMode={true}>{"x_r = \\cos(r) \\cdot x_0 + \\sin(r) \\cdot z"}</Math>
        </p>
        <p>
            Substituting the expressions for <Math displayMode={false}>x_0</Math> and <Math displayMode={false}>z</Math>:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"x_r = \\cos(r) \\cdot [\\cos(t) \\cdot x_t - \\sin(t) \\cdot v_t] + \\sin(r) \\cdot [\\sin(t) \\cdot x_t + \\cos(t) \\cdot v_t]"}</Math><br />
            <Math displayMode={true}>{"x_r = [\\cos(r)\\cos(t) + \\sin(r)\\sin(t)] \\cdot x_t + [-\\cos(r)\\sin(t) + \\sin(r)\\cos(t)] \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"x_r = \\cos(t-r) \\cdot x_t - \\sin(t-r) \\cdot v_t"}</Math>
        </p>
        <p>
            This leads to the <strong>consistency property</strong>:
        </p>
        <p class="fs-3">
            <Math displayMode={true}>{"x_r = \\cos(t-r) \\cdot x_t - \\sin(t-r) \\cdot v_t = \\cos(r) \\cdot x_0 + \\sin(r) \\cdot z"}</Math>
        </p>
        <p>
            The consistency property states that if we start from the same noise <Math displayMode={false}>z</Math> and follow the diffusion trajectory toward the data <Math displayMode={false}>x_0</Math>, 
            then <strong>any point on this trajectory should lead to the same destination</strong> <Math displayMode={false}>x_0</Math>. 
            In other words, whether we are at time <Math displayMode={false}>t</Math> (state <Math displayMode={false}>x_t</Math>) or at time <Math displayMode={false}>r</Math> (state <Math displayMode={false}>x_r</Math>), 
            both points are on the same path from <Math displayMode={false}>z</Math> to <Math displayMode={false}>x_0</Math>, and we can jump directly from one point to another without following the entire ODE trajectory.
        </p>
        
    </section>


    <section class="mb-4">
        <h2 class="mt-4 mb-3">From Consistency Property to Training Objective</h2>
        <p>
            Now we derive the training objective for consistency models. The key idea is to define a <strong>consistency function</strong> 
            <Math displayMode={false}>g_θ(x_t, t)</Math> that maps any point on the trajectory to the data <Math displayMode={false}>x_0</Math>.
        </p>
     
        <p>
            From the consistency property, we know that <Math displayMode={false}>x_0 = \cos(t) · x_t - \sin(t) · v_t</Math>. 
            We define the consistency function as:
        </p>
        <p class="fs-3">
            <Math displayMode={true}>{"g_\\theta(x_t, t) = \\cos(t) \\cdot x_t - \\sin(t) \\cdot F_\\theta(x_t, t)"}</Math>
        </p>
        <p>
            where <Math displayMode={false}>F_θ(x_t, t)</Math> is the neural network that predicts the velocity <Math displayMode={false}>v_t</Math>. 
            If the network is perfect, <Math displayMode={false}>F_θ(x_t, t) = v_t</Math>, then <Math displayMode={false}>g_θ(x_t, t) = x_0</Math>.
        </p>

        <h5 class="mt-3 mb-2">Consistency Constraint</h5>
        <p>
            The consistency property requires that for any two points <Math displayMode={false}>(x_t, t)</Math> and <Math displayMode={false}>(x_r, r)</Math> 
            on the same trajectory, they should map to the same <Math displayMode={false}>x_0</Math>:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"g_\\theta(x_t, t) = g_\\theta(x_r, r) = x_0"}</Math>
        </p>
        <p>
            This means the consistency function should be constant along the trajectory. Therefore, its total derivative along the trajectory should be zero:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = 0"}</Math>
        </p>

        <h5 class="mt-3 mb-2">Compute the Total Derivative</h5>
        <p>
            Using the chain rule, the total derivative of <Math displayMode={false}>g_θ</Math> with respect to time is:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = \\frac{\\partial g_\\theta}{\\partial x_t} \\cdot \\frac{dx_t}{dt} + \\frac{\\partial g_\\theta}{\\partial t}"}</Math>
        </p>
        <p>
            Expanding <Math displayMode={false}>g_θ = \cos(t) · x_t - \sin(t) · F_θ(x_t, t)</Math>, we compute the partial derivatives:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"\\frac{\\partial g_\\theta}{\\partial x_t} = \\cos(t) - \\sin(t) \\cdot \\frac{\\partial F_\\theta}{\\partial x_t}"}</Math><br />
            <Math displayMode={true}>{"\\frac{\\partial g_\\theta}{\\partial t} = -\\sin(t) \\cdot x_t - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{\\partial F_\\theta}{\\partial t}"}</Math>
        </p>
        <p>
            Applying the chain rule:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = \\frac{\\partial g_\\theta}{\\partial x_t} \\cdot \\frac{dx_t}{dt} + \\frac{\\partial g_\\theta}{\\partial t}"}</Math><br />
            <Math displayMode={true}>{"= \\left[\\cos(t) - \\sin(t) \\cdot \\frac{\\partial F_\\theta}{\\partial x_t}\\right] \\cdot \\frac{dx_t}{dt} + \\left[-\\sin(t) \\cdot x_t - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{\\partial F_\\theta}{\\partial t}\\right]"}</Math><br />
            <Math displayMode={true}>{"= \\cos(t) \\cdot \\frac{dx_t}{dt} - \\sin(t) \\cdot \\frac{\\partial F_\\theta}{\\partial x_t} \\cdot \\frac{dx_t}{dt} - \\sin(t) \\cdot x_t - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{\\partial F_\\theta}{\\partial t}"}</Math><br />
            <Math displayMode={true}>{"= -\\sin(t) \\cdot x_t + \\cos(t) \\cdot \\frac{dx_t}{dt} - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\left[\\frac{\\partial F_\\theta}{\\partial t} + \\frac{\\partial F_\\theta}{\\partial x_t} \\cdot \\frac{dx_t}{dt}\\right]"}</Math><br />
                    </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = -\\sin(t) \\cdot x_t + \\cos(t) \\cdot \\frac{dx_t}{dt} - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt} = 0"}</Math>

        </p>
        <p>
            where <Math displayMode={false}>{"\\frac{dF_\\theta}{dt} = \\frac{\\partial F_\\theta}{\\partial t} + \\frac{\\partial F_\\theta}{\\partial x_t} \\cdot \\frac{dx_t}{dt}"}</Math> is the total derivative of <Math displayMode={false}>F_θ</Math>.
        </p> 
        <p>
            We know that <Math displayMode={false}>{"\\frac{dx_t}{dt} = v_t = -\\sin(t) \\cdot x_0 + \\cos(t) \\cdot z"}</Math> and 
            <Math displayMode={false}>x_t = \cos(t) · x_0 + \sin(t) · z</Math>. 
            Substituting these into the expression:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = -\\sin(t) \\cdot x_t + \\cos(t) \\cdot \\frac{dx_t}{dt} - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math><br />
            <Math displayMode={true}>{"= -\\sin(t) \\cdot [\\cos(t) \\cdot x_0 + \\sin(t) \\cdot z] + \\cos(t) \\cdot [-\\sin(t) \\cdot x_0 + \\cos(t) \\cdot z] - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math>
        </p> 
        <p class="fs-7">
            <Math displayMode={true}>{"= -\\sin(t)\\cos(t) \\cdot x_0 - \\sin^2(t) \\cdot z - \\cos(t)\\sin(t) \\cdot x_0 + \\cos^2(t) \\cdot z - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math><br />
            <Math displayMode={true}>{"= [-\\sin(t)\\cos(t) - \\cos(t)\\sin(t)] \\cdot x_0 + [-\\sin^2(t) + \\cos^2(t)] \\cdot z - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math><br />
            <Math displayMode={true}>{"= -2\\sin(t)\\cos(t) \\cdot x_0 + [\\cos^2(t) - \\sin^2(t)] \\cdot z - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math>
        </p>
        <p>
            Using trigonometric identities and rearranging terms:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = -\\sin(2t) \\cdot x_0 + \\cos(2t) \\cdot z - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math>
        </p>
        <p>
            Now, we express this in terms of <Math displayMode={false}>x_t</Math> and <Math displayMode={false}>v_t</Math>. 
            From the earlier derivation, we have:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"x_0 = \\cos(t) \\cdot x_t - \\sin(t) \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"z = \\sin(t) \\cdot x_t + \\cos(t) \\cdot v_t"}</Math>
        </p>
        <p>
            Substituting into <Math displayMode={false}>-sin(2t) · x_0 + cos(2t) · z</Math>:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"-\\sin(2t) \\cdot x_0 + \\cos(2t) \\cdot z = -\\sin(2t) \\cdot [\\cos(t) \\cdot x_t - \\sin(t) \\cdot v_t] + \\cos(2t) \\cdot [\\sin(t) \\cdot x_t + \\cos(t) \\cdot v_t]"}</Math><br />
            <Math displayMode={true}>{"= -\\sin(2t)\\cos(t) \\cdot x_t + \\sin(2t)\\sin(t) \\cdot v_t + \\cos(2t)\\sin(t) \\cdot x_t + \\cos(2t)\\cos(t) \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"= [-\\sin(2t)\\cos(t) + \\cos(2t)\\sin(t)] \\cdot x_t + [\\sin(2t)\\sin(t) + \\cos(2t)\\cos(t)] \\cdot v_t"}</Math>
        </p>
        <p>
            Using trigonometric identities <Math displayMode={false}>sin(2t) = 2sin(t)cos(t)</Math> and <Math displayMode={false}>cos(2t) = cos²(t) - sin²(t)</Math>:
        </p>
        <p class="fs-7">
            <Math displayMode={true}>{"= [-2\\sin(t)\\cos^2(t) + \\sin(t)(\\cos^2(t) - \\sin^2(t))] \\cdot x_t + [2\\sin^2(t)\\cos(t) + \\cos(t)(\\cos^2(t) - \\sin^2(t))] \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"= [-2\\sin(t)\\cos^2(t) + \\sin(t)\\cos^2(t) - \\sin^3(t)] \\cdot x_t + [2\\sin^2(t)\\cos(t) + \\cos^3(t) - \\cos(t)\\sin^2(t)] \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"= [-\\sin(t)\\cos^2(t) - \\sin^3(t)] \\cdot x_t + [\\sin^2(t)\\cos(t) + \\cos^3(t)] \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"= -\\sin(t)[\\cos^2(t) + \\sin^2(t)] \\cdot x_t + \\cos(t)[\\sin^2(t) + \\cos^2(t)] \\cdot v_t"}</Math><br />
            <Math displayMode={true}>{"= -\\sin(t) \\cdot x_t + \\cos(t) \\cdot v_t"}</Math>
        </p>
        <p>
            Therefore, the complete expression becomes:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\frac{dg_\\theta}{dt} = -\\sin(t) \\cdot x_t + \\cos(t) \\cdot v_t - \\cos(t) \\cdot F_\\theta - \\sin(t) \\cdot \\frac{dF_\\theta}{dt}"}</Math><br />
            <Math displayMode={true}>{"= -\\cos(t) \\cdot (F_\\theta - v_t) - \\sin(t) \\cdot (x_t + \\frac{dF_\\theta}{dt})"}</Math>
        </p>
        <p>
            For the consistency function to remain constant along the trajectory, TrigFlow uses the cosine-scaled consistency condition:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\cos(t) \\cdot \\frac{dg_\\theta}{dt} = 0"}</Math>
        </p>
        <p>
            Substituting the expression for <Math displayMode={false}>dg_θ/dt</Math>:
        </p>
        <p class="fs-5">
            <Math displayMode={true}>{"\\cos(t) \\cdot \\frac{dg_\\theta}{dt} = -\\cos^2(t) \\cdot (F_\\theta - v_t) - \\cos(t)\\sin(t) \\cdot (x_t + \\frac{dF_\\theta}{dt}) = 0"}</Math>
        </p>
        <p>
            The reason for multiplying by <Math displayMode={false}>cos(t)</Math> will be explained below. 
            For now, we define the <strong>tangent</strong> <Math displayMode={false}>g</Math> using the stop-gradient version <Math displayMode={false}>F_θ⁻</Math>:
        </p>
        <p class="fs-3">
            <Math displayMode={true}>{"g := -\\cos^2(t) \\cdot (F_\\theta^- - v_t) - \\cos(t)\\sin(t) \\cdot (x_t + \\frac{dF_\\theta^-}{dt})"}</Math>
        </p>
        <p>
            where <Math displayMode={false}>F_θ^-</Math> denotes the stop-gradient version (frozen parameters) used during training for stability.
        </p>
        <p>
            <strong>Why multiply by <Math displayMode={false}>cos(t)</Math>?</strong> This scaling is a design choice that serves three important purposes:
        </p>
        <ul>
            <li><strong>Numerical stability:</strong> When <Math displayMode={false}>t \to \pi/2</Math> (high noise regime), 
                <Math displayMode={false}>cos(t) \to 0</Math>, causing the tangent to naturally decay to zero. 
                Without scaling, the second term <Math displayMode={false}>-sin(t)(x_t + dF_θ/dt)</Math> remains <Math displayMode={false}>O(1)</Math>, 
                leading to exploding loss in high-noise regions.</li>
            <li><strong>Velocity error dominance:</strong> The scaled form makes the velocity error term <Math displayMode={false}>-cos²(t)(F_θ - v_t)</Math> 
                the dominant component in low-to-medium noise regions, aligning with TrigFlow's goal of unifying EDM and Flow Matching.</li>
            <li><strong>Loss scale alignment:</strong> The <Math displayMode={false}>cos(t)</Math> factor acts as a noise-dependent weighting, 
                similar to the <Math displayMode={false}>σ(t)</Math> weighting in EDM formulations, ensuring consistent loss scales across different noise levels.</li>
        </ul>
        <p>
            Note that <Math displayMode={false}>cos(t) · dg_θ/dt = 0</Math> is mathematically equivalent to <Math displayMode={false}>dg_θ/dt = 0</Math> 
            (since <Math displayMode={false}>cos(t) \neq 0</Math> for <Math displayMode={false}>t \in [0, \pi/2)</Math>), 
            but the scaled version provides better training dynamics and numerical stability.
        </p>

        <h5 class="mt-3 mb-2">Training Objective</h5>
        <p>
            The training objective is to minimize the difference between the current network prediction and the target (frozen network + tangent):
        </p>
        <p class="fs-3">
            <Math displayMode={true}>{"\\mathcal{L} = ||F_\\theta(x_t, t) - F_\\theta^-(x_t, t) - g||^2"}</Math>
        </p>
        <p>
            This ensures that the network learns to predict velocities such that the consistency function remains constant along trajectories, 
            enabling fast one-step or few-step generation from any point on the trajectory.
        </p>
    </section>
 
</div>
