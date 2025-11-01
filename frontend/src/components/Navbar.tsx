import { Sparkles } from 'lucide-react';

export default function Navbar() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <Sparkles className="h-8 w-8 text-primary-600" />
            <span className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              NEUS
            </span>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <button
              onClick={() => scrollToSection('servicios')}
              className="text-gray-700 hover:text-primary-600 transition-colors"
            >
              Servicios
            </button>
            <button
              onClick={() => scrollToSection('sectores')}
              className="text-gray-700 hover:text-primary-600 transition-colors"
            >
              Sectores
            </button>
            <button
              onClick={() => scrollToSection('por-que-neus')}
              className="text-gray-700 hover:text-primary-600 transition-colors"
            >
              ¿Por qué NEUS?
            </button>
            <button
              onClick={() => scrollToSection('contacto')}
              className="px-6 py-2 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg hover:shadow-lg transition-all"
            >
              Contacto
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
