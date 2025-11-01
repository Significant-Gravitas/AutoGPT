import { Target, Search, Lock, Award, Handshake } from 'lucide-react';

const reasons = [
  {
    icon: Target,
    title: 'Orientación a Resultados',
    description: 'No vendemos tecnología por vender. Nos enfocamos en reducir costos operativos y aumentar la eficiencia de tu negocio con métricas claras y objetivos medibles.'
  },
  {
    icon: Search,
    title: 'Entendemos Antes de Actuar',
    description: 'Realizamos un diagnóstico profundo de tus procesos actuales antes de proponer cualquier solución. Esto garantiza que automatizamos lo correcto, de la manera correcta.'
  },
  {
    icon: Lock,
    title: 'Seguridad y Control Total',
    description: 'Tus datos son tuyos. Todas nuestras soluciones garantizan que mantengas el control total de tu información, con opciones de despliegue on-premise si lo requieres.'
  },
  {
    icon: Award,
    title: 'Especialización por Industria',
    description: 'Conocemos las regulaciones, desafíos y mejores prácticas de tu sector. No aplicamos soluciones genéricas, sino adaptadas a tu realidad empresarial.'
  },
  {
    icon: Handshake,
    title: 'Acompañamiento Integral',
    description: 'Desde la capacitación inicial hasta el soporte continuo, estamos contigo en cada etapa. El éxito de tu transformación digital es nuestro compromiso.'
  }
];

export default function WhyNeus() {
  return (
    <section id="por-que-neus" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            ¿Por qué elegir{' '}
            <span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              NEUS?
            </span>
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Somos más que un proveedor de tecnología, somos tu socio estratégico en transformación digital
          </p>
        </div>

        {/* Reasons List */}
        <div className="space-y-8">
          {reasons.map((reason, index) => {
            const Icon = reason.icon;
            return (
              <div
                key={index}
                className="group flex flex-col md:flex-row items-start gap-6 p-8 bg-gradient-to-r from-gray-50 to-white rounded-2xl hover:shadow-xl transition-all duration-300 border-2 border-transparent hover:border-primary-200"
              >
                {/* Number & Icon */}
                <div className="flex-shrink-0">
                  <div className="relative">
                    <div className="w-20 h-20 bg-gradient-to-r from-primary-600 to-secondary-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <Icon className="h-10 w-10 text-white" />
                    </div>
                    <div className="absolute -top-2 -right-2 w-8 h-8 bg-white border-2 border-primary-600 rounded-full flex items-center justify-center text-sm font-bold text-primary-600">
                      {index + 1}
                    </div>
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-900 mb-3 group-hover:text-primary-600 transition-colors">
                    {reason.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed text-lg">
                    {reason.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <div className="bg-gradient-to-r from-primary-600 to-secondary-600 p-1 rounded-2xl inline-block">
            <div className="bg-white p-8 rounded-2xl">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                ¿Listo para transformar tu negocio?
              </h3>
              <p className="text-gray-600 mb-6">
                Agenda un diagnóstico gratuito y descubre cómo podemos ayudarte
              </p>
              <button
                onClick={() => {
                  const element = document.getElementById('contacto');
                  element?.scrollIntoView({ behavior: 'smooth' });
                }}
                className="px-8 py-4 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg font-semibold hover:shadow-lg hover:scale-105 transition-all duration-200"
              >
                Agenda tu diagnóstico gratuito
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
