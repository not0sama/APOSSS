
interface GlowCardProps {
  children?: HTMLElement | string;
  className?: string;
  glowColor?: 'blue' | 'purple' | 'green' | 'red' | 'orange';
  size?: 'sm' | 'md' | 'lg';
  width?: string | number;
  height?: string | number;
  customSize?: boolean;
}

const glowColorMap = {
  blue: { base: 220, spread: 200 },
  purple: { base: 280, spread: 300 },
  green: { base: 120, spread: 200 },
  red: { base: 0, spread: 200 },
  orange: { base: 30, spread: 200 }
};

const sizeMap = {
  sm: 'w-48 h-64',
  md: 'w-64 h-80',
  lg: 'w-80 h-96'
};

class GlowCard {
  private cardElement: HTMLDivElement;
  private innerElement: HTMLDivElement;
  private props: GlowCardProps;
  private pointerMoveHandler: (e: PointerEvent) => void;

  constructor(props: GlowCardProps = {}) {
    this.props = {
      className: '',
      glowColor: 'blue',
      size: 'md',
      customSize: false,
      ...props
    };

    this.cardElement = document.createElement('div');
    this.innerElement = document.createElement('div');
    
    this.pointerMoveHandler = this.syncPointer.bind(this);
    this.init();
  }

  private init(): void {
    this.setupStyles();
    this.setupElements();
    this.attachEventListeners();
  }

  private setupStyles(): void {
    const beforeAfterStyles = `
      [data-glow]::before,
      [data-glow]::after {
        pointer-events: none;
        content: "";
        position: absolute;
        inset: calc(var(--border-size) * -1);
        border: var(--border-size) solid transparent;
        border-radius: calc(var(--radius) * 1px);
        background-attachment: fixed;
        background-size: calc(100% + (2 * var(--border-size))) calc(100% + (2 * var(--border-size)));
        background-repeat: no-repeat;
        background-position: 50% 50%;
        mask: linear-gradient(transparent, transparent), linear-gradient(white, white);
        mask-clip: padding-box, border-box;
        mask-composite: intersect;
      }
      
      [data-glow]::before {
        background-image: radial-gradient(
          calc(var(--spotlight-size) * 0.75) calc(var(--spotlight-size) * 0.75) at
          calc(var(--x, 0) * 1px)
          calc(var(--y, 0) * 1px),
          hsl(var(--hue, 210) calc(var(--saturation, 100) * 1%) calc(var(--lightness, 50) * 1%) / var(--border-spot-opacity, 1)), transparent 100%
        );
        filter: brightness(2);
      }
      
      [data-glow]::after {
        background-image: radial-gradient(
          calc(var(--spotlight-size) * 0.5) calc(var(--spotlight-size) * 0.5) at
          calc(var(--x, 0) * 1px)
          calc(var(--y, 0) * 1px),
          hsl(0 100% 100% / var(--border-light-opacity, 1)), transparent 100%
        );
      }
      
      [data-glow] [data-glow] {
        position: absolute;
        inset: 0;
        will-change: filter;
        opacity: var(--outer, 1);
        border-radius: calc(var(--radius) * 1px);
        border-width: calc(var(--border-size) * 20);
        filter: blur(calc(var(--border-size) * 10));
        background: none;
        pointer-events: none;
        border: none;
      }
      
      [data-glow] > [data-glow]::before {
        inset: -10px;
        border-width: 10px;
      }
    `;

    let styleElement = document.getElementById('glow-card-styles');
    if (!styleElement) {
      styleElement = document.createElement('style');
      styleElement.id = 'glow-card-styles';
      styleElement.innerHTML = beforeAfterStyles;
      document.head.appendChild(styleElement);
    }
  }

  private setupElements(): void {
    const { base, spread } = glowColorMap[this.props.glowColor!];
    
    // Setup inner element
    this.innerElement.setAttribute('data-glow', '');
    
    // Setup card element
    this.cardElement.setAttribute('data-glow', '');
    // Clear existing children before appending innerElement to prevent duplicates on updateProps
    while (this.cardElement.firstChild) {
      this.cardElement.removeChild(this.cardElement.firstChild);
    }
    this.cardElement.appendChild(this.innerElement);
    
    // Apply CSS custom properties
    const cssProps = {
      '--base': base.toString(),
      '--spread': spread.toString(),
      '--radius': '14',
      '--border': '3',
      '--backdrop': 'hsl(0 0% 60% / 0.12)',
      '--backup-border': 'var(--backdrop)',
      '--size': '200',
      '--outer': '1',
      '--border-size': 'calc(var(--border, 2) * 1px)',
      '--spotlight-size': 'calc(var(--size, 150) * 1px)',
      '--hue': 'calc(var(--base) + (var(--xp, 0) * var(--spread, 0)))',
    };

    Object.entries(cssProps).forEach(([prop, value]) => {
      this.cardElement.style.setProperty(prop, value);
    });

    // Apply background styles
    this.cardElement.style.backgroundImage = `radial-gradient(
      var(--spotlight-size) var(--spotlight-size) at
      calc(var(--x, 0) * 1px)
      calc(var(--y, 0) * 1px),
      hsl(var(--hue, 210) calc(var(--saturation, 100) * 1%) calc(var(--lightness, 70) * 1%) / var(--bg-spot-opacity, 0.1)), transparent
    )`;
    this.cardElement.style.backgroundColor = 'var(--backdrop, transparent)';
    this.cardElement.style.backgroundSize = 'calc(100% + (2 * var(--border-size))) calc(100% + (2 * var(--border-size)))';
    this.cardElement.style.backgroundPosition = '50% 50%';
    this.cardElement.style.backgroundAttachment = 'fixed';
    this.cardElement.style.border = 'var(--border-size) solid var(--backup-border)';
    this.cardElement.style.position = 'relative';
    this.cardElement.style.touchAction = 'none';

    // Apply sizing
    this.applySizing();
    
    // Apply classes
    this.applyClasses();

    // Add children
    if (this.props.children) {
      if (typeof this.props.children === 'string') {
        // Create a temporary div to parse HTML string and append its children
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = this.props.children;
        Array.from(tempDiv.children).forEach(child => {
          this.cardElement.appendChild(child.cloneNode(true));
        });
      } else {
        this.cardElement.appendChild(this.props.children);
      }
    }
  }

  private applySizing(): void {
    // Reset width/height styles before applying new ones to handle updates
    this.cardElement.style.width = '';
    this.cardElement.style.height = '';

    if (this.props.width !== undefined) {
      const width = typeof this.props.width === 'number' ? `${this.props.width}px` : this.props.width;
      this.cardElement.style.width = width;
    }
    
    if (this.props.height !== undefined) {
      const height = typeof this.props.height === 'number' ? `${this.props.height}px` : this.props.height;
      this.cardElement.style.height = height;
    }
  }

  private applyClasses(): void {
    const classes = [
      'rounded-2xl',
      'relative',
      'grid',
      'grid-rows-[1fr_auto]',
      'shadow-[0_1rem_2rem_-1rem_black]',
      'p-4',
      'gap-4',
      'backdrop-blur-[5px]'
    ];

    if (!this.props.customSize) {
      classes.push(sizeMap[this.props.size!]);
      classes.push('aspect-[3/4]');
    } else {
      // If customSize is true, ensure default size classes are removed
      Object.values(sizeMap).forEach(sizeClass => {
        const index = classes.indexOf(sizeClass);
        if (index > -1) {
          classes.splice(index, 1);
        }
      });
      const aspectIndex = classes.indexOf('aspect-[3/4]');
      if (aspectIndex > -1) {
        classes.splice(aspectIndex, 1);
      }
    }

    if (this.props.className) {
      // Split existing className to avoid issues with multiple spaces
      const customClasses = this.props.className.split(/\s+/).filter(Boolean);
      classes.push(...customClasses);
    }

    // Filter out duplicate classes
    this.cardElement.className = Array.from(new Set(classes)).join(' ');
  }

  private syncPointer(e: PointerEvent): void {
    const { clientX: x, clientY: y } = e;
    
    this.cardElement.style.setProperty('--x', x.toFixed(2));
    this.cardElement.style.setProperty('--xp', (x / window.innerWidth).toFixed(2));
    this.cardElement.style.setProperty('--y', y.toFixed(2));
    this.cardElement.style.setProperty('--yp', (y / window.innerHeight).toFixed(2));
  }

  private attachEventListeners(): void {
    // Ensure event listener is only attached once
    document.removeEventListener('pointermove', this.pointerMoveHandler);
    document.addEventListener('pointermove', this.pointerMoveHandler);
  }

  public destroy(): void {
    document.removeEventListener('pointermove', this.pointerMoveHandler);
    if (this.cardElement.parentNode) {
      this.cardElement.parentNode.removeChild(this.cardElement);
    }
  }

  public getElement(): HTMLDivElement {
    return this.cardElement;
  }

  public updateProps(newProps: Partial<GlowCardProps>): void {
    // Merge new props with existing ones
    this.props = { ...this.props, ...newProps };
    // Re-setup elements to apply all changes
    this.setupElements();
  }
}

// Usage example
function createDemo(): HTMLDivElement {
  const container = document.createElement('div');
  container.className = 'w-screen h-screen flex flex-row items-center justify-center gap-10';
  container.style.cssText = 'width: 100vw; height: 100vh; display: flex; flex-direction: row; align-items: center; justify-content: center; gap: 2.5rem;';

  const card1 = new GlowCard();
  const card2 = new GlowCard({ glowColor: 'purple' });
  const card3 = new GlowCard({ glowColor: 'green' });

  container.appendChild(card1.getElement());
  container.appendChild(card2.getElement());
  container.appendChild(card3.getElement());

  return container;
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    document.body.appendChild(createDemo());
  });
} else {
  document.body.appendChild(createDemo());
}

export default createDemo;
