import { GraduationCap, TrendingUp, Bot, Shield } from 'lucide-react';

const services = [
  {
    icon: GraduationCap,
    title: 'Capacitación en IA',
    description: 'Formamos a tus equipos para que aprovechen al máximo las herramientas de IA, desde conceptos básicos hasta implementaciones avanzadas.',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    icon: TrendingUp,
    title: 'Consultoría Estratégica',
    description: 'Analizamos tu operación actual e identificamos oportunidades de automatización que generen el mayor impacto en tus resultados.',
    color: 'from-purple-500 to-pink-500'
  },
  {
    icon: Bot,
    title: 'Desarrollo y Automatización',
    description: 'Creamos chatbots inteligentes, modelos de IA personalizados y automatizamos procesos repetitivos para liberar a tu equipo.',
    color: 'from-primary-500 to-blue-600'
  },
  {
    icon: Shield,
    title: 'Infraestructura y Seguridad',
    description: 'Implementamos soluciones seguras y escalables, garantizando el control total de tus datos y cumplimiento normativo.',
    color: 'from-green-500 to-emerald-600'
  }
];

export default function Services() {
  return (
    <section id="servicios" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Nuestros{' '}
            <span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              Servicios
            </span>
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Soluciones integrales de IA adaptadas a las necesidades específicas de tu empresa
          </p>
        </div>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {services.map((service, index) => {
            const Icon = service.icon;
            return (
              <div
                key={index}
                className="group relative p-8 bg-white border-2 border-gray-200 rounded-2xl hover:border-transparent hover:shadow-2xl transition-all duration-300"
              >
                {/* Gradient border on hover */}
                <div className={`absolute inset-0 bg-gradient-to-r ${service.color} rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10`}
                     style={{ padding: '2px' }}>
                  <div className="w-full h-full bg-white rounded-2xl"></div>
                </div>

                {/* Icon */}
                <div className={`w-16 h-16 bg-gradient-to-r ${service.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <Icon className="h-8 w-8 text-white" />
                </div>

                {/* Content */}
                <h3 className="text-2xl font-bold text-gray-900 mb-4">
                  {service.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {service.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Bottom CTA */}
        <div className="mt-12 text-center">
          <p className="text-gray-600 mb-4">
            ¿No estás seguro de qué servicio necesitas?
          </p>
          <button
            onClick={() => {
              const element = document.getElementById('contacto');
              element?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="px-6 py-3 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg font-semibold hover:shadow-lg hover:scale-105 transition-all duration-200"
          >
            Habla con un experto
          </button>
        </div>
      </div>
    </section>
  );
}
