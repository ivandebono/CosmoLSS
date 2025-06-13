import camb
from camb import model, initialpower
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numpy.fft import fftn, ifftn, fftfreq
from scipy.interpolate import interp1d

from matplotlib import colors




def get_index(array,value):

    return np.where(np.isclose(array, value))[0][0]


class PKEvolution:

    def __init__(self):
        # Set up parameters
        self.pars = camb.CAMBparams()

        As = 2e-9  # fiducial amplitude guess to start with
        # Include massive neutrinos
        self.pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, 
                        omk=0, tau=0.06, neutrino_hierarchy='normal')


        self.pars.set_matter_power(redshifts=[0.0], kmax=4.0)
        self.results = camb.get_results(self.pars)
        
        s8_fid = self.results.get_sigma8_0()
        # now set correct As using As \propto sigma8**2.
        sigma8 = 0.81  # value we want
        self.pars.InitPower.set_params(As=As * sigma8**2 / s8_fid**2, ns=0.965,r=0)
  
        
        self.pars.WantTransfer = True
        self.pars.Transfer.redshifts = list(np.linspace(0,5,100))
        self.pars.Transfer.kmax = 5.0
        self.pars.Transfer.want_cl_transfer = False  # skip C_l transfer functions
        self.pars.Transfer.include_massive_neutrinos = True  # required to get neutrino TF

        # Set the redshift range and k range for P(k,z)
        self.pars.set_matter_power(redshifts=np.linspace(0,5,100),kmax=5.0,nonlinear=True,accurate_massive_neutrino_transfers=True)
        self.results = camb.get_results(self.pars)



    def mk_results(self):

        # Calculate results
        self.results = camb.get_results(self.pars)

        return self

    def mk_pkz(self):
        # Calculate matter power spectrum 
        self.kh,self.z,self.pk=self.results.get_matter_power_spectrum(minkh=1e-4,maxkh=5,npoints=200)

        return self
    
    def plt_pk(self,z,grid=True,figsize=(10,6)):
        z_index=get_index(self.z, z)  # Find index of z

        fig,ax=plt.subplots(figsize=figsize)
        # Plot
        plt.loglog(self.kh, self.pk[z_index])
        plt.xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
        plt.ylabel(r"$P(k)$ [$h^{-3}$ Mpc$^3$]")
        plt.title(f'Matter Power Spectrum at $z = {z}$')
        if grid:
            plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.show()

        return fig


    def pk_animation(self, fps=10, filename=None,dpi=150,figsize=(10,6),grid=True,display=True):
        """Animates P(k) over z, with optional fixed axes."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set fixed axes limits (with 10% padding)
        x_min, x_max = np.min(self.kh), np.max(self.kh)
        y_min, y_max = np.min(self.pk), np.max(self.pk)
        padding = 0.1
        ax.set_xlim(x_min * (1 - padding), x_max * (1 + padding))
        ax.set_ylim(y_min * (1 - padding), y_max * (1 + padding))
        # Pre-compute z indices and axis limits (if fixed). We reverse the list so the animation goes forward in time,
        # i.e. a smaller redshigft is more recent
        z_indices = [get_index(self.z, z) for z in self.z][::-1]
   
        

        # Pre-initialize plot elements
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
        ax.set_ylabel(r"$P(k)$ [$h^{-3}$ Mpc$^3$]")
        if grid:
            ax.grid(True, which="both", ls="--", alpha=0.3)

            # Create empty line and title
        line, = ax.plot([], [], lw=2)  # Empty line
        text = ax.text(0.95, 0.95, '', transform=ax.transAxes,
               fontsize=12, ha='right', va='top')

        title="Non-linear matter power spectrum evolution"
        ax.set_title(title)
        
        
        def init():
            """Initialize animation (blank line)."""
            line.set_data([], [])
            text.set_text('')
            return line, text

    
        def update(frame):
            """Update line data and title for each z."""
            z_index = z_indices[frame]
            z=self.z[z_index]
            
            # Update line data (no axis rescaling)
            line.set_data(self.kh, self.pk[z_index])

            # Update text
            text.set_text(f"Redshift: $z$ = {z:.2f}")
        
            
            return line, text
        
        # Create animation
        ani = FuncAnimation(
        fig, update, frames=len(self.z),
        init_func=init, blit=True, interval=1000/fps)
    
        # Output handling
        if filename:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
            plt.close(fig)
            print(f"Animation saved to {filename}")
        if display:
            plt.close(fig)
            return HTML(ani.to_jshtml())
        else:
            return ani





    def generate_density_field(self,N, L, z=0):
        # Generate k-grid (in h/Mpc)
        kx = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        ky = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        kz = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        kx_3d, ky_3d, kz_3d = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)  # shape: NxNxN

        # Find the index of z
        z_index=get_index(self.z,z)
        # Interpolate P(k) onto 3D grid
        Pk_interp = interp1d(self.kh, self.pk[z_index], bounds_error=False, fill_value=0)
        Pk_3d = Pk_interp(k_mag)  # shape: NxNxN

        # Generate Gaussian random field
        rand = np.random.normal(0, 1, (N, N, N)) + 1j * np.random.normal(0, 1, (N, N, N))
        delta_k = rand * np.sqrt(Pk_3d)  # Scale by sqrt(P(k))

        # Inverse FFT to get δ(x)
        delta_x = np.fft.ifftn(delta_k).real
        return delta_x



    def plot_density_field(self, z=0, L=None, N=None,
                        cmap='viridis', vmin=None, vmax=None,
                        title=None, filename=None,dpi=200):
        """
        Plot slices of the 3D density contrast field with redshift context.
        
        Parameters:
            delta_x (ndarray): 3D density contrast field (NxNxN)
            target_z (float): Target redshift for plot annotation (uses self.z)
            L (float): Box size in Mpc/h (for axis labels)
            cmap (str): Matplotlib colormap (default: 'viridis')
            vmin/vmax (float): Colorbar limits (default: ±3σ of δ(x))
            title (str): Custom title (auto-generated if None)
            filename (str): If provided, saves plot to this path
            dpi (int): DPI for saved animation
        """

        delta_x=self.generate_density_field(N,L,z)
        
        
        # Set dynamic title if not provided
        if title is None:
            z_label = f"$z = ${z:.2f}"
            title = f"Density Contrast $\delta(x)$ at {z_label}"
        
        # Set default vmin/vmax to ±3σ if not specified
        if vmin is None or vmax is None:
            std = delta_x.std()
            vmin, vmax = -3*std, 3*std
        
        # Create figure with 3 orthogonal slices

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cbar_ax = fig.add_subplot(gs[0, 3])
            
        # Plot slices through center
        slices = {
            'XY': delta_x[:, :, N//2],
            'XZ': delta_x[:, N//2, :],
            'YZ': delta_x[N//2, :, :]
        }
        
        for ax, (label, slice_data) in zip(axes, slices.items()):
            im = ax.imshow(slice_data, cmap=cmap, 
                        norm=colors.Normalize(vmin=vmin, vmax=vmax),
                        extent=(0, L or N, 0, L or N), 
                        origin='lower')
            ax.set_title(f'{label} Plane')
            ax.set_xlabel('Mpc/h' if L else 'Grid Units')
            ax.set_ylabel('Mpc/h' if L else 'Grid Units')

        #   Add colorbar to the dedicated axis
        fig.colorbar(im, cax=cbar_ax, label='δ(x)')
        fig.suptitle(title, y=1.05)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.show()

        return self
    
    
    def mk_delta_x_z(self,N=128,L=100.0):

        return list(map(lambda z_i: self.generate_density_field(N,L,z=z_i),self.z))
    


    def create_density_evolution_animation(self, delta_x_list, redshifts, L=None, 
                                        cmap='viridis', filename='density_evolution.mp4', 
                                        fps=10, dpi=150,display=True):
        """
        Create an animation of density contrast evolution across redshifts.
        
        Parameters:
            delta_x_list (list): List of 3D density fields (one per redshift)
            redshifts (array): Corresponding redshift values
            L (float): Box size in Mpc/h (for axis labels)
            cmap (str): Colormap to use
            filename (str): Path to save the animation
            fps (int): Frames per second
            dpi (int): DPI for saved animation
        """

        plt.rcParams['animation.embed_limit'] = 100  # Set to 100MB (default is 20MB)
        
        # Determine global vmin/vmax across all redshifts
        global_min = min([d.min() for d in delta_x_list])
        global_max = max([d.max() for d in delta_x_list])
        std = np.mean([d.std() for d in delta_x_list])
        vmin, vmax = -3*std, 3*std  # Fixed scale based on average std
        
        # Set up figure with consistent colorbar
        fig = plt.figure(figsize=(15, 5.5))  # Increased height slightly
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], top=0.85)  # Added top parameter
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        cbar_ax = fig.add_subplot(gs[0, 3])



        fig.suptitle('Density Contrast Evolution in the $\Lambda$CDM Universe', y=0.95)  # Lower y-position
        
        # Initialize plots
        ims = []
        for ax in axes:
            im = ax.imshow(np.zeros_like(delta_x_list[0][:,:,0]), 
                        cmap=cmap,
                        norm=colors.Normalize(vmin=vmin, vmax=vmax),
                        extent=(0, L or delta_x_list[0].shape[0], 
                                0, L or delta_x_list[0].shape[0]),
                        origin='lower')
            ims.append(im)
            ax.set_xlabel('Mpc $h^{-1}$' if L else 'Grid Units')

        # Only put the y-label on the leftmost subplot 
        axes[0].set_ylabel('Mpc $h^{-1}$' if L else 'Grid Units')
        
        # Set titles 
        axes[0].set_title('XY Plane')
        axes[1].set_title('XZ Plane')
        axes[2].set_title('YZ Plane')
        
        # Add colorbar
        fig.colorbar(ims[0], cax=cbar_ax, label='$\delta(x)$')
        fig.tight_layout()

        # Create text object
        text = fig.text(0.5, 0.02, '', ha='center')

        def init():
            text.set_text('')  # Initialize empty
            return text, ims[0], ims[1], ims[2]

        
        # Animation update function
        def update(frame):
            # Show the evolution in chronological order, i.e. put the largest redshift first, and zero last.
            delta_x = delta_x_list[::-1][frame]
            z = redshifts[::-1][frame]
            
            # Update slice data
            ims[0].set_array(delta_x[:, :, delta_x.shape[2]//2])
            ims[1].set_array(delta_x[:, delta_x.shape[1]//2, :])
            ims[2].set_array(delta_x[delta_x.shape[0]//2, :, :])
            
            # Update text
            text.set_text(f"Redshift: $z$ = {z:.2f}")

            return text, ims[0], ims[1], ims[2]
        
        # Create animation
        ani = FuncAnimation(fig, update, init_func=init,frames=len(delta_x_list),
                        interval=500/fps, blit=False)
        
            
        # Output handling
        if filename:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
            plt.close(fig)
            print(f"Animation saved to {filename}")
        if display:
            plt.close(fig)
            return HTML(ani.to_jshtml())
        else:
            return ani
